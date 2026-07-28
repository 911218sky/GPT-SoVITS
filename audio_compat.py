from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

AudioPath = Union[str, Path]


def load_audio(path: AudioPath) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    waveform = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1).contiguous()
    return waveform, int(sample_rate)


class AudioResample(torch.nn.Module):
    def __init__(self, orig_freq: int, new_freq: int) -> None:
        super().__init__()
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)
        divisor = math.gcd(self.orig_freq, self.new_freq)
        self.up = self.new_freq // divisor
        self.down = self.orig_freq // divisor

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[-1] == 0 or self.orig_freq == self.new_freq:
            return waveform
        original_dtype = waveform.dtype
        cpu_waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
        samples = cpu_waveform.numpy()
        resampled = resample_poly(samples, self.up, self.down, axis=-1)
        output = torch.from_numpy(np.ascontiguousarray(resampled))
        return output.to(device=waveform.device, dtype=original_dtype)
