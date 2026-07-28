from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import BinaryIO

import numpy as np
import soundfile as sf
import torch

from audio_compat import AudioResample

AudioSource = str | Path | BinaryIO | BytesIO


def load(
    source: AudioSource,
    normalize: bool = True,
    channels_first: bool = True,
) -> tuple[torch.Tensor, int]:
    del normalize
    samples, sample_rate = sf.read(
        source,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32).T)
    if not channels_first:
        waveform = waveform.transpose(0, 1).contiguous()
    return waveform, int(sample_rate)


def _resample(
    waveform: torch.Tensor,
    orig_freq: int,
    new_freq: int,
) -> torch.Tensor:
    return AudioResample(orig_freq, new_freq)(waveform)


def patch(module: ModuleType) -> None:
    transforms_module = getattr(module, "transforms")
    functional_module = getattr(module, "functional")

    setattr(module, "load", load)
    setattr(transforms_module, "Resample", AudioResample)
    setattr(functional_module, "resample", _resample)


def install() -> None:
    torchaudio_module = ModuleType("torchaudio")
    transforms_module = ModuleType("torchaudio.transforms")
    functional_module = ModuleType("torchaudio.functional")

    setattr(transforms_module, "Resample", AudioResample)
    setattr(functional_module, "resample", _resample)
    setattr(torchaudio_module, "__version__", "compat")
    setattr(torchaudio_module, "load", load)
    setattr(torchaudio_module, "transforms", transforms_module)
    setattr(torchaudio_module, "functional", functional_module)

    sys.modules["torchaudio"] = torchaudio_module
    sys.modules["torchaudio.transforms"] = transforms_module
    sys.modules["torchaudio.functional"] = functional_module
