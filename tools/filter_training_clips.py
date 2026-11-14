#!/usr/bin/env python3
"""Filter training clips by duration, quiet-high noise, music bleed, CJK text."""
import argparse, json, re, shutil
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

CJK = re.compile(r"[\u4e00-\u9fff]")

def metrics(path, sr=32000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    dur = len(y) / sr
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rms = librosa.feature.rms(y=y, hop_length=256)[0]
    quiet = rms <= np.percentile(rms, 25)
    high = (freqs >= 6000) & (freqs < 12000)
    mid = (freqs >= 200) & (freqs < 4000)
    qh = float(S[high][:, quiet].mean()) if quiet.any() else 0.0
    qm = float(S[mid][:, quiet].mean()) if quiet.any() else 0.0
    sm = float(S[mid][:, ~quiet].mean()) if (~quiet).any() else 1.0
    bleed = qm / (sm + 1e-12)
    # SNR proxy
    frames = []
    hop = sr // 100
    win = hop * 4
    for i in range(0, max(1, len(y) - win), hop):
        frames.append(np.sqrt(np.mean(y[i : i + win] ** 2) + 1e-12))
    frames = np.array(frames) if frames else np.array([1e-6])
    snr = 20 * np.log10((np.percentile(frames, 75) + 1e-12) / (np.percentile(frames, 10) + 1e-12))
    return {"dur": dur, "quiet_high": qh, "bleed": bleed, "snr": float(snr), "rms": float(np.sqrt(np.mean(y**2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--list-in", required=True, help="slicer.list path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--list-out", required=True)
    ap.add_argument("--min-sec", type=float, default=2.8)
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--max-quiet-high", type=float, default=0.12)
    ap.add_argument("--max-bleed", type=float, default=0.045)
    ap.add_argument("--min-snr", type=float, default=12.0)
    ap.add_argument("--min-cjk", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], []
    lines = Path(args.list_in).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        # format: wav_path|speaker|lang|text   OR wav|text
        parts = line.split("|")
        wav = parts[0]
        text = parts[-1] if len(parts) >= 2 else ""
        p = Path(wav)
        if not p.exists():
            # try relative to wav-dir
            p2 = Path(args.wav_dir) / Path(wav).name
            p = p2 if p2.exists() else p
        if not p.exists():
            dropped.append({"file": wav, "reason": "missing"})
            continue
        m = metrics(str(p))
        cjk_n = len(CJK.findall(text))
        reason = None
        if m["dur"] < args.min_sec or m["dur"] > args.max_sec:
            reason = f"dur={m['dur']:.2f}"
        elif m["quiet_high"] > args.max_quiet_high:
            reason = f"quiet_high={m['quiet_high']:.4f}"
        elif m["bleed"] > args.max_bleed:
            reason = f"bleed={m['bleed']:.4f}"
        elif m["snr"] < args.min_snr:
            reason = f"snr={m['snr']:.1f}"
        elif cjk_n < args.min_cjk:
            reason = f"cjk={cjk_n}"
        if reason:
            dropped.append({"file": p.name, "reason": reason, **m})
            continue
        dest = out / p.name
        shutil.copy2(p, dest)
        # rewrite list path to dest
        if len(parts) >= 4:
            new_line = f"{dest}|{parts[1]}|{parts[2]}|{parts[3]}"
        elif len(parts) == 2:
            new_line = f"{dest}|{parts[1]}"
        else:
            new_line = f"{dest}|{'|'.join(parts[1:])}"
        kept.append(new_line)

    Path(args.list_out).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    summary = {"kept": len(kept), "dropped": len(dropped), "dropped_samples": dropped[:20]}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    Path(out / "filter_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
