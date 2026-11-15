import os

import torch
import torch.nn.functional as F
from module.models import CFM


V5_TEMPERATURE = 0.875
V5_REFERENCE_ANCHOR = 1.0
V5_ROLLING_TAIL_FRAMES = 32
V5_MAX_TARGET_CHUNK_FRAMES = 640
V5_REFERENCE_FRAMES = 500
V5_TOTAL_CHUNK_FRAMES = 1000
V5_VERSIONS = {"v5dev", "v5turbo"}


class CFMV5(CFM):
    def __init__(self, in_channels, dit):
        super().__init__(in_channels, dit)
        self.use_step_embedding = False
        self.cfg_drop_text = False
        self.noise_temperature = V5_TEMPERATURE
        self.use_static_cache = os.environ.get(
            "GSVV5_CFM_STATIC_CACHE", os.environ.get("GSVV4_CFM_STATIC_CACHE", "1")
        ) != "0"
        # v5 only. Equal-length batches can skip the dense attention mask.
        self.estimator.drop_dense_mask = True

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, n_timesteps, temperature=V5_TEMPERATURE,
                  inference_cfg_rate=1.30, generator=None):
        return super().inference(mu, x_lens, prompt, n_timesteps, temperature, inference_cfg_rate, generator=generator)

    def forward(self, x1, x_lens, prompt_lens, mu, use_grad_ckpt):
        b = x1.shape[0]
        t = torch.rand([b], device=mu.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1, device=mu.device)
        vt = x1 - x0
        xt = x0 + t[:, None, None] * vt
        prompt = torch.zeros_like(x1)
        for i in range(b):
            prompt[i, :, : prompt_lens[i]] = x1[i, :, : prompt_lens[i]]
            xt[i, :, : prompt_lens[i]] = 0
        vt_pred = self.estimator(
            xt, prompt, x_lens, t, text0=mu, use_grad_ckpt=use_grad_ckpt,
        ).transpose(2, 1)
        loss = 0
        for i in range(b):
            loss += self.criterion(vt_pred[i, :, prompt_lens[i] : x_lens[i]], vt[i, :, prompt_lens[i] : x_lens[i]])
        return loss / b


@torch.inference_mode()
def synthesize_v5_mel(model, reference_features, target_features, reference_mel,
                      sample_steps=None, cfg_rate=None):
    steps = (4 if model.version == "v5turbo" else 32) if sample_steps is None else int(sample_steps)
    cfg = (1.30 if model.version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
    reference_frames = min(reference_mel.shape[-1], reference_features.shape[-1])
    reference_mel = reference_mel[..., :reference_frames][..., -_v5_reference_limit():]
    reference_features = reference_features[..., :reference_frames][..., -_v5_reference_limit():]
    reference_frames = reference_mel.shape[-1]
    chunk_frames = min(V5_TOTAL_CHUNK_FRAMES - reference_frames, V5_MAX_TARGET_CHUNK_FRAMES)
    original_mel = reference_mel
    original_features = reference_features
    rolling_mel = reference_mel
    rolling_features = reference_features
    results = []
    for index, start in enumerate(range(0, target_features.shape[-1], chunk_frames)):
        target = target_features[..., start:start + chunk_frames]
        if index == 0:
            prompt = original_mel
            features = original_features
        else:
            tail = min(V5_ROLLING_TAIL_FRAMES, reference_frames,
                       rolling_mel.shape[-1], rolling_features.shape[-1])
            prefix = reference_frames - tail
            prompt = torch.cat((original_mel[..., :prefix], rolling_mel[..., -tail:]), dim=-1)
            features = torch.cat((original_features[..., :prefix], rolling_features[..., -tail:]), dim=-1)
        mu = torch.cat((features, target), dim=-1).transpose(2, 1)
        lengths = torch.full((mu.shape[0],), mu.shape[1], dtype=torch.long, device=mu.device)
        generated = model.cfm.inference(
            mu, lengths, prompt, steps, inference_cfg_rate=cfg,
        )[..., prompt.shape[-1]:]
        results.append(generated)
        rolling_mel = generated[..., -reference_frames:].to(reference_mel.dtype)
        rolling_features = target[..., -reference_frames:]
    return torch.cat(results, dim=-1)



def _v5_cfm_batch():
    try:
        return max(1, int(os.environ.get("GPT_SOVITS_V5_CFM_BATCH", "16")))
    except ValueError:
        return 16


def _v5_reference_limit():
    """Last N mel frames of the reference. 10 ms each. Shorter sequences are faster."""
    try:
        return max(32, int(os.environ.get("GPT_SOVITS_V5_REF_FRAMES", "160")))
    except ValueError:
        return 160


def _infer_v5_group(model, reference_features, reference_mel, reference_frames, group, steps, cfg):
    """One CFM call for several single-chunk sentences. Same prompt, masked tails."""
    if not group:
        return {}
    if len(group) == 1:
        index, target = group[0]
        mel = synthesize_v5_mel(
            model, reference_features, target, reference_mel, steps, cfg,
        )
        return {index: mel}
    try:
        lengths = [int(target.shape[-1]) for _, target in group]
        width = max(lengths)
        padded = []
        for _, target in group:
            gap = width - int(target.shape[-1])
            padded.append(target if gap == 0 else F.pad(target, (0, gap)))
        target_batch = torch.cat(padded, 0)
        feature_batch = reference_features.expand(target_batch.shape[0], -1, -1)
        mu = torch.cat((feature_batch, target_batch), dim=-1).transpose(2, 1)
        x_lens = torch.tensor(
            [reference_frames + length for length in lengths],
            device=mu.device, dtype=torch.long,
        )
        generated = model.cfm.inference(
            mu, x_lens, reference_mel, steps, inference_cfg_rate=cfg,
        )
        return {
            index: generated[row:row + 1, :, reference_frames:reference_frames + length]
            for row, ((index, _), length) in enumerate(zip(group, lengths))
        }
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(group) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(group) // 2
        left = _infer_v5_group(
            model, reference_features, reference_mel, reference_frames, group[:middle], steps, cfg,
        )
        right = _infer_v5_group(
            model, reference_features, reference_mel, reference_frames, group[middle:], steps, cfg,
        )
        left.update(right)
        return left


@torch.inference_mode()
def synthesize_v5_mels(model, reference_features, targets, reference_mel,
                       sample_steps=None, cfg_rate=None, max_batch=None):
    """Batch v5 CFM across sentences. Long sentences keep the rolling single path."""
    steps = (4 if model.version == "v5turbo" else 32) if sample_steps is None else int(sample_steps)
    cfg = (1.30 if model.version == "v5dev" else 0.0) if cfg_rate is None else float(cfg_rate)
    if max_batch is None:
        max_batch = _v5_cfm_batch()
    reference_frames = min(reference_mel.shape[-1], reference_features.shape[-1])
    reference_mel = reference_mel[..., :reference_frames][..., -_v5_reference_limit():]
    reference_features = reference_features[..., :reference_frames][..., -_v5_reference_limit():]
    reference_frames = reference_mel.shape[-1]
    chunk_frames = min(V5_TOTAL_CHUNK_FRAMES - reference_frames, V5_MAX_TARGET_CHUNK_FRAMES)
    outputs = [None] * len(targets)
    shorts = []
    for index, target in enumerate(targets):
        if int(target.shape[-1]) > chunk_frames:
            outputs[index] = synthesize_v5_mel(
                model, reference_features, target, reference_mel, steps, cfg,
            )
        else:
            shorts.append((index, target))
    shorts.sort(key=lambda item: int(item[1].shape[-1]))
    for start in range(0, len(shorts), max_batch):
        done = _infer_v5_group(
            model, reference_features, reference_mel, reference_frames,
            shorts[start:start + max_batch], steps, cfg,
        )
        for index, mel in done.items():
            outputs[index] = mel
    return outputs
