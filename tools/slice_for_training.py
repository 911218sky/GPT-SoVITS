#!/usr/bin/env python3
"""Slice speech for GPT-SoVITS training.

Algorithm (community + official):
1. openvpi/slicer2 silence cuts (same as tools/slicer2.py / WebUI)
2. Any clip longer than max_length_sec is recursively split at the
   quietest RMS frame (RVC Dataset Maker / common TTS practice).
   GPT-SoVITS issue #1070: prefer ~3–10s; >10s often unused in training.
3. Drop clips shorter than min_length_sec.

Usage (from repo root):
  .venv/bin/python tools/slice_for_training.py \\
    --input path/to/speech.wav --out-dir path/to/slicer \\
    [--threshold -34] [--min-length-ms 4000] [--max-length-sec 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from slicer2 import Slicer  # noqa: E402
from tools.my_utils import load_audio  # noqa: E402


def rms_frames(x: np.ndarray, hop: int, win: int) -> np.ndarray:
    if len(x) < win:
        return np.array([float(np.sqrt(np.mean(x * x) + 1e-12))])
    n = 1 + (len(x) - win) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = x[i * hop : i * hop + win]
        out[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    return out


def split_at_quietest(audio: np.ndarray, sr: int, max_sec: float, min_sec: float) -> list[tuple[int, int]]:
    """Return sample index ranges within [0, len(audio))."""
    n = len(audio)
    max_n = int(max_sec * sr)
    min_n = int(min_sec * sr)
    if n <= max_n:
        return [(0, n)] if n >= min_n else []

    hop = max(1, sr // 100)  # 10ms
    win = hop * 4
    rms = rms_frames(audio, hop, win)

    # avoid cutting near edges
    edge = max(1, int(0.8 * sr / hop))
    search = rms[edge : max(edge + 1, len(rms) - edge)]
    if len(search) == 0:
        mid = n // 2
        left = split_at_quietest(audio[:mid], sr, max_sec, min_sec)
        right = [(a + mid, b + mid) for a, b in split_at_quietest(audio[mid:], sr, max_sec, min_sec)]
        return left + right

    cut_frame = int(np.argmin(search)) + edge
    cut = min(n - 1, cut_frame * hop)
    # ensure both sides have some length
    if cut < min_n // 2 or n - cut < min_n // 2:
        cut = n // 2

    left = split_at_quietest(audio[:cut], sr, max_sec, min_sec)
    right = [(a + cut, b + cut) for a, b in split_at_quietest(audio[cut:], sr, max_sec, min_sec)]
    return left + right


def normalize_chunk(chunk: np.ndarray, _max: float = 0.9, alpha: float = 0.25) -> np.ndarray:
    peak = float(np.abs(chunk).max())
    if peak < 1e-8:
        return chunk
    if peak > 1:
        chunk = chunk / peak
        peak = 1.0
    return (chunk / peak * (_max * alpha)) + (1 - alpha) * chunk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="wav file or directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--threshold", type=float, default=-34.0, help="dB silence threshold (WebUI default -34)")
    ap.add_argument("--min-length-ms", type=int, default=4000, help="slicer2 min_length ms")
    ap.add_argument("--min-interval-ms", type=int, default=300)
    ap.add_argument("--hop-size-ms", type=int, default=10)
    ap.add_argument("--max-sil-kept-ms", type=int, default=500)
    ap.add_argument("--max-length-sec", type=float, default=10.0, help="force-split above this (issue #1070)")
    ap.add_argument("--min-keep-sec", type=float, default=2.5, help="drop shorter final clips")
    ap.add_argument("--max-keep-sec", type=float, default=12.0, help="drop longer final clips after split")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.wav"):
        old.unlink()

    files = [inp] if inp.is_file() else sorted(p for p in inp.iterdir() if p.suffix.lower() in {".wav", ".flac", ".mp3"})
    slicer = Slicer(
        sr=args.sr,
        threshold=args.threshold,
        min_length=args.min_length_ms,
        min_interval=args.min_interval_ms,
        hop_size=args.hop_size_ms,
        max_sil_kept=args.max_sil_kept_ms,
    )

    meta = []
    idx = 0
    for f in files:
        audio = load_audio(str(f), args.sr)
        # slicer2 yields (chunk, start_frame, end_frame) in some forks; GPT-SoVITS yields chunks only or triples
        pieces = list(slicer.slice(audio))
        raw_ranges: list[tuple[np.ndarray, int, int]] = []
        # Compatible with both return styles
        cursor = 0
        for item in pieces:
            if isinstance(item, (tuple, list)) and len(item) == 3:
                chunk, start, end = item
                raw_ranges.append((np.asarray(chunk, dtype=np.float32), int(start), int(end)))
            else:
                chunk = np.asarray(item, dtype=np.float32)
                # approximate positions by scanning (fallback)
                raw_ranges.append((chunk, cursor, cursor + len(chunk)))
                cursor += len(chunk)

        for chunk, _s, _e in raw_ranges:
            for a, b in split_at_quietest(chunk, args.sr, args.max_length_sec, args.min_keep_sec):
                sub = chunk[a:b]
                dur = len(sub) / args.sr
                if dur < args.min_keep_sec or dur > args.max_keep_sec:
                    continue
                # skip near-silent
                if float(np.sqrt(np.mean(sub * sub) + 1e-12)) < 0.005:
                    continue
                sub = normalize_chunk(sub)
                name = f"{args.prefix}{f.stem}_{idx:04d}_{dur*1000:07.0f}ms.wav"
                path = out / name
                wavfile.write(str(path), args.sr, (np.clip(sub, -1, 1) * 32767).astype(np.int16))
                meta.append({"file": name, "dur": round(dur, 3), "source": f.name})
                idx += 1

    (out / "slice_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    durs = [m["dur"] for m in meta]
    if not durs:
        print("ERROR: no clips produced")
        sys.exit(1)
    print(
        f"wrote {len(durs)} clips -> {out}\n"
        f"total_min={sum(durs)/60:.2f} mean={sum(durs)/len(durs):.2f}s "
        f"min={min(durs):.2f} max={max(durs):.2f}"
    )


if __name__ == "__main__":
    main()
