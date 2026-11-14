"""Select optional AR acceleration for the requested device and precision."""

from functools import lru_cache
from importlib import import_module
import logging
import os
from pathlib import Path
import sys

import torch


_logger = logging.getLogger(__name__)
_DLL_HANDLES = []


def _cuda_available(device):
    return device.type == "cuda" and torch.cuda.is_available() and torch.version.cuda is not None


def _low_vram_gpu(device, threshold_gb=10.0):
    """RTX 3060 Ti 8GB: CUDA Graph static buffers starve VITS and collapse throughput."""
    try:
        props = torch.cuda.get_device_properties(device)
        return (props.total_memory / (1024 ** 3)) <= float(threshold_gb)
    except Exception:
        return False


def cuda_graph_available(device):
    device = torch.device(device)
    # CUDAGraph env: 0=off, 1=force on, unset/auto=on only when >10GB
    flag = os.environ.get("CUDAGraph", "auto").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    force_on = flag in {"1", "true", "on", "yes"}
    if not force_on and _cuda_available(device) and _low_vram_gpu(device):
        return False
    return bool(
        _cuda_available(device)
        and callable(getattr(torch.cuda, "CUDAGraph", None))
        and callable(getattr(torch.cuda, "graph", None))
        and callable(getattr(torch.cuda, "Stream", None))
        and callable(getattr(torch.nn.functional, "scaled_dot_product_attention", None))
    )


@lru_cache(maxsize=1)
def _flash_attention_module():
    if os.name == "nt":
        root = Path(__file__).resolve().parents[1]
        for directory in (
            root / "GPT_SoVITS" / "Accel" / "vendor" / "cuda" / "bin",
            Path(sys.executable).resolve().parent / "Lib" / "site-packages" / "torch" / "lib",
        ):
            if directory.is_dir():
                _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    try:
        return import_module("flash_attn")
    except (ImportError, OSError, RuntimeError) as exc:
        _logger.info("FlashAttention is unavailable: %s", exc)
        return None


def flash_attention_available(device, dtype):
    device = torch.device(device)
    if not _cuda_available(device) or dtype not in (torch.float16, torch.bfloat16):
        return False
    if torch.cuda.get_device_capability(device)[0] < 8:
        return False
    module = _flash_attention_module()
    return callable(getattr(module, "flash_attn_with_kvcache", None))


def resolve_acceleration(device, dtype, use_cuda_graph=True, use_flash_attention=True):
    graph = bool(use_cuda_graph and cuda_graph_available(device))
    if use_flash_attention and flash_attention_available(device, dtype):
        return "flash_attn_varlen_cuda_graph", graph
    if graph:
        return "torch_static_cuda_graph", True
    return None, False


def create_acceleration(
    weights_path,
    device,
    dtype,
    max_batch_size=32,
    use_cuda_graph=True,
    use_flash_attention=True,
    max_seq_length=None,
):
    device = torch.device(device)
    # Forced CUDA Graph on 8GB must stay small or VITS thrashers.
    if use_cuda_graph and _cuda_available(device) and _low_vram_gpu(device):
        try:
            cap = int(os.environ.get("GPT_SOVITS_T2S_MAX_BATCH", "8"))
        except ValueError:
            cap = 8
        max_batch_size = min(int(max_batch_size), max(1, cap))
    backend, enabled = resolve_acceleration(device, dtype, use_cuda_graph, use_flash_attention)
    if backend is None:
        if use_cuda_graph:
            _logger.info(
                "AR CUDA Graph disabled (low VRAM or unavailable); using ordinary T2S inference"
            )
        return None
    try:
        from GPT_SoVITS.Accel.adapter import AccelInference
        import_module("GPT_SoVITS.Accel.PyTorch.AR.backends." + backend)
    except (ImportError, OSError, RuntimeError) as exc:
        _logger.warning("AR acceleration is unavailable; using ordinary inference: %s", exc)
        return None
    _logger.info(
        "AR acceleration backend=%s max_batch_size=%s cuda_graph=%s",
        backend,
        max_batch_size,
        enabled,
    )
    return AccelInference(
        weights_path,
        device=device,
        dtype=dtype,
        max_batch_size=max_batch_size,
        use_cuda_graph=use_cuda_graph,
        use_flash_attention=use_flash_attention,
        max_seq_length=max_seq_length,
    )
