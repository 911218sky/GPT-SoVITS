from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import TypedDict

import requests
from tqdm import tqdm

from .common import ROLE_PROFILES, RoleProfile, get_role_profile, require_file

LOGGER = logging.getLogger(__name__)


DEFAULT_SERVER_URL = "http://127.0.0.1:9880"
DEFAULT_MAX_RETRIES = 3
# 3060 Ti 8GB 掃速結果（2026-09）：bs=64 + len=1200 約比舊 bs=56/len=2400 快 ~30%
DEFAULT_MAX_TEXT_LENGTH = 1200
DEFAULT_BATCH_SIZE = 64
DEFAULT_BATCH_THRESHOLD = 0.75
DEFAULT_FRAGMENT_INTERVAL = 0.01
DEFAULT_REPETITION_PENALTY = 1.35
DEFAULT_SPLIT_METHOD = "cut5"
DEFAULT_TOP_K = 15
# 本機最快組 + 微優化（skip empty_cache / TQDM_DISABLE / SV cache）確認平均約 328 字/秒
DEFAULT_CHARS_PER_SEC = 328.0


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}小時{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def estimate_chars_per_sec(chunk_chars: list[int], chunk_seconds: list[float], fallback: float) -> float:
    if not chunk_seconds or not chunk_chars:
        return fallback
    total_chars = sum(chunk_chars)
    total_sec = sum(chunk_seconds)
    if total_sec <= 0 or total_chars <= 0:
        return fallback
    return total_chars / total_sec


class TTSRequestPayload(TypedDict):
    text: str
    text_lang: str
    ref_audio_path: str
    aux_ref_audio_paths: list[str]
    prompt_text: str
    prompt_lang: str
    top_k: int
    top_p: float
    temperature: float
    text_split_method: str
    batch_size: int
    batch_threshold: float
    split_bucket: bool
    speed_factor: float
    fragment_interval: float
    seed: int
    media_type: str
    streaming_mode: bool
    parallel_infer: bool
    repetition_penalty: float


def filter_text(text: str) -> str:
    circle_numbers = "①②③④⑤⑥⑦⑧⑨⑩"
    text = re.sub(
        r"[①②③④⑤⑥⑦⑧⑨⑩]",
        lambda match: str(circle_numbers.index(match.group()) + 1),
        text,
    )

    def num_to_chinese(match: re.Match[str]) -> str:
        numeric_text = match.group()
        digits = "零一二三四五六七八九"
        if len(numeric_text) > 9:
            return "".join(digits[int(digit)] for digit in numeric_text)

        number = int(numeric_text)
        if number == 0:
            return "零"
        units = ["", "十", "百", "千", "萬", "十萬", "百萬", "千萬", "億"]
        result = ""
        position = 0
        while number > 0:
            value = number % 10
            if value:
                result = digits[value] + units[position] + result
            elif result and not result.startswith("零"):
                result = "零" + result
            number //= 10
            position += 1
        return result.replace("一十", "十")

    text = re.sub(r"\d+", num_to_chinese, text)
    filtered = re.sub(r"[^\u4e00-\u9fa5a-zA-Z，。！？、；,.]", "", text)
    return ",".join(filtered.split())


def split_text(text: str, max_text_length: int) -> list[str]:
    if max_text_length < 1:
        raise ValueError("--max-text-length 必須大於 0")
    delimiters = "。！？；，、"
    chunks: list[str] = []
    current = ""
    for character in text:
        current += character
        if len(current) < max_text_length:
            continue
        split_point = max(
            (current.rfind(delimiter) for delimiter in delimiters if current.rfind(delimiter) > len(current) // 2),
            default=-1,
        )
        split_point = split_point + 1 if split_point >= 0 else max_text_length
        chunks.append(current[:split_point])
        current = current[split_point:]
    if current:
        chunks.append(current)
    return chunks


def build_payload(profile: RoleProfile, text: str, args: argparse.Namespace) -> TTSRequestPayload:
    speed_factor = args.speed_factor if args.speed_factor is not None else profile.speed_factor
    return {
        "text": filter_text(text),
        "text_lang": args.text_lang,
        "ref_audio_path": str(require_file(profile.ref_audio_path, "參考音訊")),
        "aux_ref_audio_paths": [],
        "prompt_text": profile.prompt_text,
        "prompt_lang": profile.prompt_lang,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "text_split_method": args.split_method,
        "batch_size": args.batch_size,
        "batch_threshold": args.batch_threshold,
        "split_bucket": args.split_bucket,
        "speed_factor": speed_factor,
        "fragment_interval": args.fragment_interval,
        "seed": args.seed,
        "media_type": args.media_type,
        "streaming_mode": False,
        "parallel_infer": args.parallel_infer,
        "repetition_penalty": args.repetition_penalty,
    }


def set_model(server_url: str, profile: RoleProfile) -> None:
    for endpoint, path in (
        ("/set_gpt_weights", profile.gpt_weights_path),
        ("/set_sovits_weights", profile.sovits_weights_path),
    ):
        response = requests.get(
            f"{server_url}{endpoint}",
            params={"weights_path": str(require_file(path, "模型權重"))},
            timeout=900,
        )
        response.raise_for_status()


def request_audio(server_url: str, payload: TTSRequestPayload, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{server_url}/tts",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=900,
            )
            response.raise_for_status()
            return response.content
        except (requests.RequestException, OSError) as error:
            last_error = error
            if attempt < retries:
                LOGGER.warning("TTS 失敗，重試 %d/%d：%s", attempt + 1, retries, error)
    raise RuntimeError(f"TTS 請求失敗：{last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-SoVITS 批次 TTS 產生工具")
    parser.add_argument("--file-path", type=Path, required=True, help="輸入文字檔")
    parser.add_argument("--role", choices=sorted(ROLE_PROFILES), default="真人男")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--split-method", choices=["cut1", "cut2", "cut3", "cut4", "cut5"], default=DEFAULT_SPLIT_METHOD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--batch-threshold", type=float, default=DEFAULT_BATCH_THRESHOLD)
    parser.add_argument("--split-bucket", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--parallel-infer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text-lang", default="zh")
    parser.add_argument("--speed-factor", type=float, default=None)
    parser.add_argument("--fragment-interval", type=float, default=DEFAULT_FRAGMENT_INTERVAL)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--media-type", choices=["wav", "ogg", "aac"], default="wav")
    parser.add_argument("--set-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--chars-per-sec",
        type=float,
        default=DEFAULT_CHARS_PER_SEC,
        help="預估吞吐（字/秒），用於開跑時顯示整本 ETA；跑起來後會用實測值更新",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = get_role_profile(args.role)
    input_path = require_file(args.file_path, "輸入文字檔")
    output_dir = args.output_dir or Path(__file__).resolve().parent / "output" / f"GPT_{args.role}_{input_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.set_model:
        set_model(args.server_url.rstrip("/"), profile)

    text = input_path.read_text(encoding="utf-8", errors="ignore").replace("\n", "").replace(" ", "")
    chunks = split_text(text, args.max_text_length)
    extension = args.media_type
    total_chars = len(text)
    pending_indices = [
        index
        for index, _chunk in enumerate(chunks)
        if not (output_dir / f"{index}.{extension}").exists()
    ]
    pending_chars = sum(len(chunks[index]) for index in pending_indices)
    rate = args.chars_per_sec if args.chars_per_sec > 0 else DEFAULT_CHARS_PER_SEC
    eta0 = pending_chars / rate if rate > 0 else 0.0
    LOGGER.info(
        "批次參數：chunks=%d, pending=%d, max_text_length=%d, split_method=%s, "
        "batch_size=%d, split_bucket=%s, parallel_infer=%s",
        len(chunks),
        len(pending_indices),
        args.max_text_length,
        args.split_method,
        args.batch_size,
        args.split_bucket,
        args.parallel_infer,
    )
    LOGGER.info(
        "小說規模：總字數≈%d，待轉換≈%d 字 / %d 段；依 %.0f 字/秒預估約需 %s",
        total_chars,
        pending_chars,
        len(pending_indices),
        rate,
        format_duration(eta0),
    )
    chunk_seconds: list[float] = []
    chunk_chars_done: list[int] = []
    skipped = 0
    done_chars = 0
    wall_started = time.perf_counter()
    for index, chunk in enumerate(tqdm(chunks, desc=str(output_dir))):
        output_path = output_dir / f"{index}.{extension}"
        if output_path.exists():
            skipped += 1
            continue
        payload = build_payload(profile, chunk, args)
        if not payload["text"]:
            LOGGER.warning("第 %d 段過濾後為空，跳過", index)
            continue
        started = time.perf_counter()
        output_path.write_bytes(request_audio(args.server_url.rstrip("/"), payload, args.max_retries))
        elapsed = time.perf_counter() - started
        chars = len(payload["text"])
        chunk_seconds.append(elapsed)
        chunk_chars_done.append(chars)
        done_chars += chars
        live_rate = estimate_chars_per_sec(chunk_chars_done, chunk_seconds, rate)
        remain_chars = max(0, pending_chars - done_chars)
        remain_eta = remain_chars / live_rate if live_rate > 0 else 0.0
        elapsed_wall = time.perf_counter() - wall_started
        LOGGER.info(
            "chunk=%d chars=%d client_wall=%.3fs (chars/s=%.1f) | "
            "進度 %d/%d 段，實測≈%.0f 字/秒，已用 %s，預估剩餘 %s",
            index,
            chars,
            elapsed,
            chars / elapsed if elapsed > 0 else 0.0,
            len(chunk_seconds),
            len(pending_indices),
            live_rate,
            format_duration(elapsed_wall),
            format_duration(remain_eta),
        )
    if chunk_seconds:
        total = sum(chunk_seconds)
        final_rate = estimate_chars_per_sec(chunk_chars_done, chunk_seconds, rate)
        LOGGER.info(
            "client_summary: done=%d skipped=%d total=%.3fs avg=%.3fs min=%.3fs max=%.3fs | "
            "整本實測≈%.0f 字/秒，總耗時 %s",
            len(chunk_seconds),
            skipped,
            total,
            total / len(chunk_seconds),
            min(chunk_seconds),
            max(chunk_seconds),
            final_rate,
            format_duration(time.perf_counter() - wall_started),
        )
        LOGGER.info(
            "server stage breakdown is in API logs / GPT_SOVITS_TIMING_LOG ([TTS_TIMING] lines)"
        )
    elif skipped:
        LOGGER.info("全部區段已存在，已跳過 %d 段，無需轉換", skipped)
    LOGGER.info("完成：%s", output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
