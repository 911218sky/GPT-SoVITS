from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from .batch_tts import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_THRESHOLD,
    DEFAULT_FRAGMENT_INTERVAL,
    DEFAULT_MAX_TEXT_LENGTH,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SPLIT_METHOD,
    DEFAULT_TOP_K,
    build_payload,
    split_text,
)
from .common import get_role_profile, require_file

LOGGER = logging.getLogger(__name__)
DEFAULT_SERVER_URL = "http://127.0.0.1:9880"


@dataclass(frozen=True)
class SweepConfig:
    name: str
    batch_size: int = DEFAULT_BATCH_SIZE
    max_text_length: int = DEFAULT_MAX_TEXT_LENGTH
    batch_threshold: float = DEFAULT_BATCH_THRESHOLD
    split_bucket: bool = True
    parallel_infer: bool = True
    top_k: int = DEFAULT_TOP_K
    split_method: str = DEFAULT_SPLIT_METHOD
    fragment_interval: float = DEFAULT_FRAGMENT_INTERVAL
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY


def request_tts(server_url: str, payload: dict[str, Any], timeout: float = 900.0) -> bytes:
    response = requests.post(
        f"{server_url}/tts",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.content


def config_to_namespace(cfg: SweepConfig) -> argparse.Namespace:
    return argparse.Namespace(
        text_lang="zh",
        top_k=cfg.top_k,
        top_p=1.0,
        temperature=1.0,
        split_method=cfg.split_method,
        batch_size=cfg.batch_size,
        batch_threshold=cfg.batch_threshold,
        split_bucket=cfg.split_bucket,
        speed_factor=None,
        fragment_interval=cfg.fragment_interval,
        seed=42,
        media_type="wav",
        parallel_infer=cfg.parallel_infer,
        repetition_penalty=cfg.repetition_penalty,
    )


def run_config(
    server_url: str,
    profile,
    raw_text: str,
    cfg: SweepConfig,
) -> dict[str, Any]:
    args = config_to_namespace(cfg)
    chunks = split_text(raw_text, cfg.max_text_length)
    started = time.perf_counter()
    bytes_out = 0
    chunk_seconds: list[float] = []
    try:
        for index, chunk in enumerate(chunks):
            payload = build_payload(profile, chunk, args)
            payload["pipeline_prefetch"] = False
            if not payload["text"]:
                continue
            t0 = time.perf_counter()
            audio = request_tts(server_url, payload)
            elapsed = time.perf_counter() - t0
            chunk_seconds.append(elapsed)
            bytes_out += len(audio)
        wall = time.perf_counter() - started
        chars = len(raw_text)
        return {
            "ok": True,
            "name": cfg.name,
            "config": asdict(cfg),
            "chunks": len(chunk_seconds),
            "wall_s": wall,
            "chars_per_s": chars / wall if wall > 0 else 0.0,
            "chunk_avg_s": statistics.mean(chunk_seconds) if chunk_seconds else 0.0,
            "bytes_out": bytes_out,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - sweep must continue on OOM/HTTP errors
        wall = time.perf_counter() - started
        return {
            "ok": False,
            "name": cfg.name,
            "config": asdict(cfg),
            "chunks": len(chunk_seconds),
            "wall_s": wall,
            "chars_per_s": 0.0,
            "chunk_avg_s": statistics.mean(chunk_seconds) if chunk_seconds else 0.0,
            "bytes_out": bytes_out,
            "error": f"{type(error).__name__}: {error}",
        }


def build_phase1() -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for batch_size in (32, 40, 48, 56, 64, 72):
        for max_len in (1200, 1600, 2000, 2400):
            configs.append(
                SweepConfig(
                    name=f"p1_bs{batch_size}_len{max_len}",
                    batch_size=batch_size,
                    max_text_length=max_len,
                )
            )
    return configs


def build_phase2(base: SweepConfig) -> list[SweepConfig]:
    base_fields = {k: v for k, v in asdict(base).items() if k != "name"}
    configs = [
        SweepConfig(name="p2_baseline_top", **base_fields),
        SweepConfig(name="p2_no_bucket", **{**base_fields, "split_bucket": False}),
        SweepConfig(name="p2_no_parallel", **{**base_fields, "parallel_infer": False}),
        SweepConfig(name="p2_topk5", **{**base_fields, "top_k": 5}),
        SweepConfig(name="p2_topk25", **{**base_fields, "top_k": 25}),
        SweepConfig(name="p2_thr0.60", **{**base_fields, "batch_threshold": 0.60}),
        SweepConfig(name="p2_thr0.90", **{**base_fields, "batch_threshold": 0.90}),
        SweepConfig(name="p2_cut2", **{**base_fields, "split_method": "cut2"}),
        SweepConfig(name="p2_frag0.01", **{**base_fields, "fragment_interval": 0.01}),
        SweepConfig(name="p2_frag0.00", **{**base_fields, "fragment_interval": 0.0}),
    ]
    for batch_size in sorted({base.batch_size - 8, base.batch_size, base.batch_size + 8}):
        if batch_size < 16:
            continue
        for max_len in sorted({base.max_text_length - 400, base.max_text_length, base.max_text_length + 400}):
            if max_len < 800:
                continue
            configs.append(
                SweepConfig(
                    name=f"p2_fine_bs{batch_size}_len{max_len}",
                    **{**base_fields, "batch_size": batch_size, "max_text_length": max_len},
                )
            )
    seen: set[str] = set()
    unique: list[SweepConfig] = []
    for cfg in configs:
        if cfg.name in seen:
            continue
        seen.add(cfg.name)
        unique.append(cfg)
    return unique


def pick_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [row for row in results if row.get("ok")]
    if not ok:
        return None
    return max(ok, key=lambda row: row["chars_per_s"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-SoVITS 批次參數掃速")
    parser.add_argument("--file-path", type=Path, required=True)
    parser.add_argument("--role", default="真人男")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("local_tts/output/_bench_sweep"))
    parser.add_argument("--repeats", type=int, default=2, help="最終確認重複次數")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_role_profile(args.role)
    server_url = args.server_url.rstrip("/")
    raw = require_file(args.file_path, "輸入文字檔").read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\n", "").replace(" ", "")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("warmup...")
    warm = SweepConfig(name="warmup", batch_size=40, max_text_length=800)
    run_config(server_url, profile, raw[:500], warm)

    all_results: list[dict[str, Any]] = []

    LOGGER.info("phase1: batch_size x max_text_length (%d configs)", len(build_phase1()))
    phase1_results: list[dict[str, Any]] = []
    for cfg in build_phase1():
        LOGGER.info("run %s", cfg.name)
        row = run_config(server_url, profile, raw, cfg)
        phase1_results.append(row)
        all_results.append(row)
        status = f"{row['chars_per_s']:.1f} c/s wall={row['wall_s']:.2f}s" if row["ok"] else row["error"]
        LOGGER.info("  -> %s", status)

    best1 = pick_best(phase1_results)
    if best1 is None:
        LOGGER.error("phase1 全部失敗")
        (out_dir / "results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    base = SweepConfig(**best1["config"])
    LOGGER.info("phase1 best: %s (%.1f c/s)", base.name, best1["chars_per_s"])

    phase2 = build_phase2(base)
    LOGGER.info("phase2: refinements (%d configs)", len(phase2))
    phase2_results: list[dict[str, Any]] = []
    for cfg in phase2:
        LOGGER.info("run %s", cfg.name)
        row = run_config(server_url, profile, raw, cfg)
        phase2_results.append(row)
        all_results.append(row)
        status = f"{row['chars_per_s']:.1f} c/s wall={row['wall_s']:.2f}s" if row["ok"] else row["error"]
        LOGGER.info("  -> %s", status)

    best2 = pick_best(phase2_results) or best1
    winner_cfg = SweepConfig(**best2["config"])
    LOGGER.info("phase2 best: %s (%.1f c/s)", winner_cfg.name, best2["chars_per_s"])

    confirm_rows: list[dict[str, Any]] = []
    winner_fields = {k: v for k, v in asdict(winner_cfg).items() if k != "name"}
    for index in range(args.repeats):
        cfg = SweepConfig(name=f"confirm_{index+1}", **winner_fields)
        LOGGER.info("confirm %s", cfg.name)
        row = run_config(server_url, profile, raw, cfg)
        confirm_rows.append(row)
        all_results.append(row)

    ok_confirm = [row for row in confirm_rows if row["ok"]]
    confirm_avg = statistics.mean(row["chars_per_s"] for row in ok_confirm) if ok_confirm else 0.0
    confirm_wall = statistics.mean(row["wall_s"] for row in ok_confirm) if ok_confirm else 0.0

    # Compare against previous defaults
    default_cfg = SweepConfig(name="default_ref")
    default_row = run_config(server_url, profile, raw, default_cfg)
    all_results.append(default_row)

    summary = {
        "input_chars": len(raw),
        "role": args.role,
        "phase1_best": best1,
        "phase2_best": best2,
        "winner_config": asdict(winner_cfg),
        "confirm_avg_chars_per_s": confirm_avg,
        "confirm_avg_wall_s": confirm_wall,
        "default_ref": default_row,
        "speedup_vs_default": (
            (confirm_avg / default_row["chars_per_s"] - 1.0) if default_row.get("ok") and confirm_avg else None
        ),
    }
    (out_dir / "results.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "winner.json").write_text(json.dumps(asdict(winner_cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    ranked = sorted((row for row in all_results if row.get("ok")), key=lambda row: row["chars_per_s"], reverse=True)
    print("\n======== SWEEP TOP 10 ========")
    for row in ranked[:10]:
        cfg = row["config"]
        print(
            f"{row['chars_per_s']:7.1f} c/s  wall={row['wall_s']:6.2f}s  "
            f"bs={cfg['batch_size']:<3} len={cfg['max_text_length']:<4} "
            f"bucket={cfg['split_bucket']} par={cfg['parallel_infer']} "
            f"top_k={cfg['top_k']} thr={cfg['batch_threshold']} "
            f"cut={cfg['split_method']} frag={cfg['fragment_interval']}  [{row['name']}]"
        )
    print("-------- WINNER --------")
    print(json.dumps(asdict(winner_cfg), ensure_ascii=False, indent=2))
    if default_row.get("ok"):
        print(
            f"default: {default_row['chars_per_s']:.1f} c/s | "
            f"winner confirm avg: {confirm_avg:.1f} c/s | "
            f"speedup: {summary['speedup_vs_default']*100:+.1f}%"
            if summary["speedup_vs_default"] is not None
            else "n/a"
        )
    print("============================\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
