#!/usr/bin/env python3
"""Convert WhisperX JSON to GPT-SoVITS .list lines; optionally keep one speaker."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("json_path", type=Path)
    p.add_argument("--audio", type=Path, required=True, help="source wav/mp3 used for cutting")
    p.add_argument("--out-dir", type=Path, required=True, help="directory for cut clips + list")
    p.add_argument("--speaker", default=None, help="keep only this speaker id, e.g. SPEAKER_00")
    p.add_argument("--list-name", default="slicer.list")
    p.add_argument("--speaker-name", default="speaker", help="name written into .list column 2")
    p.add_argument("--lang", default="ZH")
    p.add_argument("--min-dur", type=float, default=0.4)
    args = p.parse_args()

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    segs = data.get("segments") or []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    list_path = args.out_dir / args.list_name
    lines: list[str] = []
    kept = 0

    for i, seg in enumerate(segs):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("speaker")
        if args.speaker and spk and spk != args.speaker:
            continue
        if args.speaker and not spk:
            # diarize missing — skip when filtering
            continue
        start = float(seg.get("start") or 0)
        end = float(seg.get("end") or 0)
        if end - start < args.min_dur:
            continue
        clip = args.out_dir / f"seg_{i:05d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(start), "-to", str(end),
                "-i", str(args.audio),
                "-ac", "1", "-ar", "32000",
                str(clip),
            ],
            check=True,
        )
        lines.append(f"{clip.resolve()}|{args.speaker_name}|{args.lang}|{text}")
        kept += 1

    list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {kept} clips -> {list_path}")


if __name__ == "__main__":
    main()
