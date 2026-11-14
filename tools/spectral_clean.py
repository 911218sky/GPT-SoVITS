#!/usr/bin/env python3
"""Mild spectral gate + highband hush for TTS training vocals.
Designed to reduce UVR residue / hiss without killing speech formants.
"""
import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


def soft_gate(y, sr, noise_pct=12, atten_db=18, attack=0.01, release=0.08):
    hop = max(1, sr // 100)
    win = hop * 4
    # frame RMS
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop)[0]
    thr = np.percentile(rms, noise_pct) * 1.6
    # expand to sample mask with smoothing
    mask = (rms > thr).astype(np.float32)
    # smooth
    k = max(1, int(release * sr / hop))
    ker = np.ones(k, dtype=np.float32) / k
    mask = np.convolve(mask, ker, mode="same")
    mask = np.clip(mask, 0, 1)
    # upsample
    m = np.interp(np.arange(len(y)), np.arange(len(mask)) * hop, mask).astype(np.float32)
    floor = 10 ** (-atten_db / 20)
    gain = floor + (1 - floor) * m
    return y * gain


def highband_hush(y, sr, cutoff=6500, max_atten_db=10):
    """Attenuate highband only in low-energy frames (hiss/bleed)."""
    n_fft = 2048
    hop = 256
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high = freqs >= cutoff
    # frame energy in speech band
    speech = (freqs >= 200) & (freqs < 4000)
    e = mag[speech].mean(axis=0)
    thr = np.percentile(e, 30)
    quiet = e <= thr
    atten = np.ones_like(mag)
    factor = 10 ** (-max_atten_db / 20)
    # stronger atten on quieter frames
    for i, q in enumerate(quiet):
        if q:
            atten[high, i] = factor
        else:
            atten[high, i] = 10 ** (-(max_atten_db * 0.25) / 20)
    mag2 = mag * atten
    y2 = librosa.istft(mag2 * np.exp(1j * phase), hop_length=hop, length=len(y))
    return y2.astype(np.float32)


def peak_normalize(y, peak=0.95):
    m = np.max(np.abs(y)) + 1e-9
    if m > peak:
        y = y * (peak / m)
    return y


def process_file(inp, out, sr_out=32000):
    y, sr = librosa.load(inp, sr=44100, mono=True)
    y = soft_gate(y, sr, noise_pct=12, atten_db=16)
    y = highband_hush(y, sr, cutoff=6500, max_atten_db=12)
    y = peak_normalize(y, 0.95)
    if sr_out != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, y, sr_out, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--sr", type=int, default=32000)
    args = ap.parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if inp.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted(inp.glob("*.wav")):
            process_file(str(p), str(out / p.name), args.sr)
            print("ok", p.name)
    else:
        process_file(str(inp), str(out), args.sr)
        print("ok", out)

if __name__ == "__main__":
    main()
