from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import TypedDict

import requests
from tqdm import tqdm

from .common import ROLE_PROFILES, RoleProfile, get_role_profile, require_file

LOGGER = logging.getLogger(__name__)


DEFAULT_SERVER_URL = "http://127.0.0.1:9880"
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TEXT_LENGTH = 20000


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
        number = int(match.group())
        if number == 0:
            return "零"
        digits = "零一二三四五六七八九"
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
    return {
        "text": filter_text(text),
        "text_lang": "zh",
        "ref_audio_path": str(require_file(profile.ref_audio_path, "參考音訊")),
        "aux_ref_audio_paths": [],
        "prompt_text": profile.prompt_text,
        "prompt_lang": profile.prompt_lang,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "text_split_method": args.split_method,
        "batch_size": args.batch_size,
        "batch_threshold": 0.75,
        "split_bucket": False,
        "speed_factor": profile.speed_factor,
        "fragment_interval": 0.01,
        "seed": -1,
        "media_type": args.media_type,
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
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
    parser.add_argument("--split-method", choices=["cut1", "cut2", "cut3", "cut4", "cut5"], default="cut2")
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--media-type", choices=["wav", "ogg", "aac"], default="wav")
    parser.add_argument("--set-model", action=argparse.BooleanOptionalAction, default=True)
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
    for index, chunk in enumerate(tqdm(chunks, desc=str(output_dir))):
        output_path = output_dir / f"{index}.{extension}"
        if output_path.exists():
            continue
        payload = build_payload(profile, chunk, args)
        if not payload["text"]:
            LOGGER.warning("第 %d 段過濾後為空，跳過", index)
            continue
        output_path.write_bytes(request_audio(args.server_url.rstrip("/"), payload, args.max_retries))
    LOGGER.info("完成：%s", output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
