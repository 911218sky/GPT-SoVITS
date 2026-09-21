from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .batch_tts import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_THRESHOLD,
    DEFAULT_FRAGMENT_INTERVAL,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SPLIT_METHOD,
    DEFAULT_TOP_K,
    build_payload,
    filter_text,
    split_text,
)
from .common import get_role_profile, require_file
from .finish_audio import remove_silence

LOGGER = logging.getLogger(__name__)
DEFAULT_SERVER_URL = "http://127.0.0.1:9880"


def request_tts(server_url: str, payload: dict, timeout: float = 900.0) -> bytes:
    response = requests.post(
        f"{server_url}/tts",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.content


def make_payload(profile, text: str, args: argparse.Namespace, pipeline_prefetch: bool) -> dict:
    payload = build_payload(profile, text, args)
    payload["pipeline_prefetch"] = pipeline_prefetch
    return payload


def run_tts_oneshot(
    server_url: str,
    profile,
    text: str,
    args: argparse.Namespace,
    output_path: Path,
    pipeline_prefetch: bool,
) -> float:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    payload = make_payload(profile, text, args, pipeline_prefetch=pipeline_prefetch)
    started = time.perf_counter()
    output_path.write_bytes(request_tts(server_url, payload))
    return time.perf_counter() - started


def run_tts_chunked(
    server_url: str,
    profile,
    chunks: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[float, list[Path]]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    started = time.perf_counter()
    for index, chunk in enumerate(chunks):
        payload = make_payload(profile, chunk, args, pipeline_prefetch=False)
        if not payload["text"]:
            continue
        path = output_dir / f"{index}.wav"
        path.write_bytes(request_tts(server_url, payload))
        paths.append(path)
    return time.perf_counter() - started, paths


def clean_files(paths: list[Path], output_dir: Path) -> float:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for path in paths:
        remove_silence(
            path,
            output_dir / f"{path.stem}.mp3",
            silence_duration=0.5,
            silence_threshold=-30,
            quality=4,
            volume_boost=1.0,
        )
    return time.perf_counter() - started


def run_tts_with_clean_pipeline(
    server_url: str,
    profile,
    chunks: list[str],
    args: argparse.Namespace,
    wav_dir: Path,
    clean_dir: Path,
) -> tuple[float, float, float]:
    """Returns (wall, tts_only_sum, clean_only_sum) with TTS∥clean overlap."""
    if wav_dir.exists():
        shutil.rmtree(wav_dir)
    if clean_dir.exists():
        shutil.rmtree(clean_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    tts_sum = 0.0
    clean_sum = 0.0
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        clean_futures = []
        for index, chunk in enumerate(chunks):
            payload = make_payload(profile, chunk, args, pipeline_prefetch=False)
            if not payload["text"]:
                continue
            wav_path = wav_dir / f"{index}.wav"
            t0 = time.perf_counter()
            wav_path.write_bytes(request_tts(server_url, payload))
            tts_sum += time.perf_counter() - t0
            mp3_path = clean_dir / f"{index}.mp3"

            def _clean(src: Path = wav_path, dst: Path = mp3_path) -> float:
                c0 = time.perf_counter()
                remove_silence(src, dst, silence_duration=0.5, silence_threshold=-30, quality=4, volume_boost=1.0)
                return time.perf_counter() - c0

            clean_futures.append(pool.submit(_clean))
        for future in clean_futures:
            clean_sum += future.result()
    wall = time.perf_counter() - wall_started
    return wall, tts_sum, clean_sum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="量測 text 預取與 TTS∥清音 的加速")
    parser.add_argument("--file-path", type=Path, required=True)
    parser.add_argument("--role", default="真人男")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("local_tts/output/_bench_overlap"))
    parser.add_argument("--max-text-length", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-threshold", type=float, default=DEFAULT_BATCH_THRESHOLD)
    parser.add_argument("--split-bucket", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-method", default=DEFAULT_SPLIT_METHOD)
    parser.add_argument("--text-lang", default="zh")
    parser.add_argument("--speed-factor", type=float, default=None)
    parser.add_argument("--fragment-interval", type=float, default=DEFAULT_FRAGMENT_INTERVAL)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--media-type", default="wav")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_role_profile(args.role)
    server_url = args.server_url.rstrip("/")
    input_path = require_file(args.file_path, "輸入文字檔")
    raw = input_path.read_text(encoding="utf-8", errors="ignore").replace("\n", "").replace(" ", "")
    chunks = split_text(raw, args.max_text_length)
    full_text = filter_text(raw)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    LOGGER.info("warmup...")
    warmup = (chunks[0] if chunks else full_text)[:400]
    run_tts_oneshot(server_url, profile, warmup, args, out / "_warmup.wav", pipeline_prefetch=False)

    LOGGER.info("A1) oneshot baseline (no prefetch), chars=%d", len(full_text))
    a1 = run_tts_oneshot(server_url, profile, raw, args, out / "A1_baseline.wav", pipeline_prefetch=False)
    LOGGER.info("A1 wall=%.3fs", a1)

    LOGGER.info("A2) oneshot + pipeline_prefetch")
    a2 = run_tts_oneshot(server_url, profile, raw, args, out / "A2_prefetch.wav", pipeline_prefetch=True)
    LOGGER.info("A2 wall=%.3fs", a2)

    LOGGER.info("B1) chunked TTS then clean (sequential), chunks=%d", len(chunks))
    b1_tts, wavs = run_tts_chunked(server_url, profile, chunks, args, out / "B1_wav")
    b1_clean = clean_files(wavs, out / "B1_clean")
    b1_wall = b1_tts + b1_clean
    LOGGER.info("B1 tts=%.3fs clean=%.3fs wall=%.3fs", b1_tts, b1_clean, b1_wall)

    LOGGER.info("B2) chunked TTS ∥ clean")
    b2_wall, b2_tts, b2_clean = run_tts_with_clean_pipeline(
        server_url, profile, chunks, args, out / "B2_wav", out / "B2_clean"
    )
    LOGGER.info("B2 tts_sum=%.3fs clean_sum=%.3fs wall=%.3fs", b2_tts, b2_clean, b2_wall)

    def pct(new: float, old: float) -> str:
        if old <= 0:
            return "n/a"
        return f"{(old - new) / old * 100:+.1f}%"

    print("\n======== OVERLAP BENCH SUMMARY ========")
    print(f"input_chars={len(raw)} chunks={len(chunks)} role={args.role}")
    print(f"A1 oneshot baseline:          {a1:.3f}s")
    print(f"A2 oneshot text-prefetch:     {a2:.3f}s   delta={pct(a2, a1)} vs A1")
    print(f"B1 TTS then clean:            {b1_wall:.3f}s  (tts={b1_tts:.3f}s clean={b1_clean:.3f}s)")
    print(f"B2 TTS∥clean:                 {b2_wall:.3f}s  (tts_sum={b2_tts:.3f}s clean_sum={b2_clean:.3f}s)")
    print(f"B2 vs B1 wall:                {pct(b2_wall, b1_wall)}")
    print("=======================================\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
