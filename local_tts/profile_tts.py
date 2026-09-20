from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests

from .common import ROLE_PROFILES, get_role_profile, require_file
from .batch_tts import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_THRESHOLD,
    DEFAULT_FRAGMENT_INTERVAL,
    DEFAULT_MAX_TEXT_LENGTH,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SERVER_URL,
    DEFAULT_SPLIT_METHOD,
    DEFAULT_TOP_K,
    build_payload,
    filter_text,
    set_model,
    split_text,
)

LOGGER = logging.getLogger(__name__)
TIMING_RE = re.compile(
    r"\[TTS_TIMING\]\s+"
    r"ref=(?P<ref>[\d.]+)s\s+"
    r"text=(?P<text>[\d.]+)s\s+"
    r"t2s=(?P<t2s>[\d.]+)s\s+"
    r"vits=(?P<vits>[\d.]+)s\s+"
    r"post=(?P<post>[\d.]+)s\s+"
    r"total=(?P<total>[\d.]+)s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="量測 GPT-SoVITS 單次請求的階段耗時（需 API 已啟動）")
    parser.add_argument("--file-path", type=Path, default=Path(__file__).resolve().parent / "examples" / "test_novel.txt")
    parser.add_argument("--role", choices=sorted(ROLE_PROFILES), default="真人男")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--max-chunks", type=int, default=2, help="最多量測幾個文字 chunk（含 warmup 後）")
    parser.add_argument("--warmup", type=int, default=1, help="先跑幾個 chunk 不計入統計")
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-threshold", type=float, default=DEFAULT_BATCH_THRESHOLD)
    parser.add_argument("--split-bucket", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-method", choices=["cut1", "cut2", "cut3", "cut4", "cut5"], default=DEFAULT_SPLIT_METHOD)
    parser.add_argument("--text-lang", default="zh")
    parser.add_argument("--speed-factor", type=float, default=None)
    parser.add_argument("--fragment-interval", type=float, default=DEFAULT_FRAGMENT_INTERVAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--media-type", choices=["wav", "ogg", "aac"], default="wav")
    parser.add_argument("--set-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--api-log",
        type=Path,
        default=None,
        help="API stdout 日誌路徑；若提供會解析 [TTS_TIMING] 行做伺服器端細分",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="可選：把量測音檔寫出",
    )
    return parser.parse_args()


def summarize(rows: list[dict[str, float]]) -> None:
    if not rows:
        print("沒有可統計的伺服器端 [TTS_TIMING] 資料。")
        return
    keys = ["ref", "text", "t2s", "vits", "post", "total"]
    averages = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    stage_sum = sum(averages[key] for key in ("ref", "text", "t2s", "vits", "post")) or 1.0
    print("")
    print(f"=== 伺服器端階段平均（n={len(rows)}）===")
    for key, label in (
        ("ref", "參考音/prompt"),
        ("text", "文本+BERT"),
        ("t2s", "T2S"),
        ("vits", "VITS合成"),
        ("post", "後處理拼接"),
        ("total", "請求內合計"),
    ):
        value = averages[key]
        if key == "total":
            print(f"  {label:12s} {value:7.3f}s")
        else:
            print(f"  {label:12s} {value:7.3f}s  ({value / stage_sum * 100:5.1f}%)")
    ranked = sorted(
        ((averages[k], name) for k, name in (("t2s", "T2S"), ("vits", "VITS"), ("text", "文本+BERT"), ("ref", "參考音"), ("post", "後處理"))),
        reverse=True,
    )
    print(f"最耗時：{ranked[0][1]}（{ranked[0][0]:.3f}s）")


def main() -> int:
    args = parse_args()
    profile = get_role_profile(args.role)
    input_path = require_file(args.file_path, "輸入文字檔")
    server_url = args.server_url.rstrip("/")
    text = input_path.read_text(encoding="utf-8", errors="ignore").replace("\n", "").replace(" ", "")
    chunks = split_text(text, args.max_text_length)
    if not chunks:
        raise RuntimeError("輸入文字切分後為空")

    if args.set_model:
        print("載入角色模型…", flush=True)
        set_model(server_url, profile)

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_offset = 0
    if args.api_log is not None and args.api_log.is_file():
        log_offset = args.api_log.stat().st_size

    client_rows: list[float] = []
    measure_indices = list(range(args.warmup, min(len(chunks), args.warmup + args.max_chunks)))
    all_indices = list(range(0, min(len(chunks), args.warmup + args.max_chunks)))
    print(
        f"共 {len(chunks)} chunks，warmup={args.warmup}，量測 index={measure_indices}",
        flush=True,
    )

    for index in all_indices:
        chunk = chunks[index]
        payload = build_payload(profile, chunk, args)
        if not payload["text"]:
            print(f"chunk {index}: 過濾後為空，跳過", flush=True)
            continue
        chars = len(filter_text(chunk))
        started = time.perf_counter()
        response = requests.post(
            f"{server_url}/tts",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            timeout=900,
        )
        response.raise_for_status()
        elapsed = time.perf_counter() - started
        if output_dir is not None:
            (output_dir / f"{index}.{args.media_type}").write_bytes(response.content)
        tag = "warmup" if index < args.warmup else "measure"
        print(
            f"[{tag}] chunk={index} chars={chars} client_http={elapsed:.3f}s audio_bytes={len(response.content)}",
            flush=True,
        )
        if index >= args.warmup:
            client_rows.append(elapsed)

    server_rows: list[dict[str, float]] = []
    if args.api_log is not None and args.api_log.is_file():
        content = args.api_log.read_text(encoding="utf-8", errors="ignore")[log_offset:]
        matches = list(TIMING_RE.finditer(content))
        # 最後 N 筆對應本次量測（含 warmup）
        keep = matches[-len(all_indices) :] if matches else []
        measure_keep = keep[args.warmup :] if len(keep) > args.warmup else keep
        for match in measure_keep:
            server_rows.append({key: float(match.group(key)) for key in ("ref", "text", "t2s", "vits", "post", "total")})
        if not server_rows:
            print(f"警告：在 {args.api_log} 找不到 [TTS_TIMING]（請確認 API 有把 stdout 導到此檔）")
        else:
            print("")
            print("=== 各次伺服器計時 ===")
            for i, row in enumerate(server_rows):
                print(
                    f"  #{i}: ref={row['ref']:.3f} text={row['text']:.3f} "
                    f"t2s={row['t2s']:.3f} vits={row['vits']:.3f} "
                    f"post={row['post']:.3f} total={row['total']:.3f}"
                )
            summarize(server_rows)

    if client_rows:
        avg = sum(client_rows) / len(client_rows)
        print("")
        print(f"=== 客戶端 HTTP 平均（n={len(client_rows)}）===")
        print(f"  端到端 {avg:.3f}s / chunk")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
