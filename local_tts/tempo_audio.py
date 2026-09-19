from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".aac", ".flac", ".m4a"})
NUMBERED_STEM = re.compile(r"^(\d+)$")


def build_atempo_filter(tempo: float) -> str:
    """把語速倍數拆成 FFmpeg atempo 鏈（每段必須在 0.5～2.0）。"""
    if tempo <= 0:
        raise ValueError("--tempo 必須大於 0")

    factors: list[float] = []
    remaining = tempo
    while remaining < 0.5 or remaining > 2.0:
        if remaining < 0.5:
            factors.append(0.5)
            remaining /= 0.5
        else:
            factors.append(2.0)
            remaining /= 2.0
    factors.append(remaining)
    return ",".join(f"atempo={factor:.10g}" for factor in factors)


def apply_tempo(
    input_file: Path,
    output_file: Path,
    tempo: float,
    quality: int = 4,
) -> None:
    """以 FFmpeg atempo 調整單一音檔語速（音高不變）。"""
    if quality < 0 or quality > 9:
        raise ValueError("quality 必須介於 0 到 9")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = build_atempo_filter(tempo)
    suffix = output_file.suffix.lower()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-af",
        filter_graph,
    ]
    if suffix == ".mp3":
        command.extend(["-c:a", "libmp3lame", "-q:a", str(quality)])
    elif suffix == ".wav":
        command.extend(["-c:a", "pcm_s16le"])
    else:
        command.extend(["-c:a", "aac" if suffix in {".aac", ".m4a"} else "libmp3lame"])
    command.append(str(output_file))

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "沒有 FFmpeg 錯誤訊息"
        raise RuntimeError(f"FFmpeg 語速調整失敗：{input_file}\n{detail}") from error


def list_audio_files(input_folder: Path, suffix: str | None, numbered_only: bool) -> list[Path]:
    if not input_folder.is_dir():
        raise NotADirectoryError(f"輸入資料夾不存在：{input_folder}")

    candidates: list[Path] = []
    if suffix:
        normalized = suffix.lstrip(".").lower()
        paths = input_folder.glob(f"*.{normalized}")
    else:
        paths = (path for path in input_folder.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)

    for path in paths:
        if not path.is_file():
            continue
        if numbered_only and not NUMBERED_STEM.fullmatch(path.stem):
            continue
        candidates.append(path)

    if numbered_only:
        return sorted(candidates, key=lambda path: (int(path.stem), path.name))
    return sorted(candidates, key=lambda path: path.name)


def process_tempo_folder(
    input_folder: Path,
    output_folder: Path,
    tempo: float,
    suffix: str | None = None,
    numbered_only: bool = True,
    max_workers: int | None = None,
    quality: int = 4,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，請先安裝並加入 PATH")

    audio_files = list_audio_files(input_folder, suffix, numbered_only)
    if not audio_files:
        LOGGER.warning("在 %s 中找不到可處理的音訊檔", input_folder)
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    worker_count = max_workers or max(1, (os.cpu_count() or 4) - 2)
    tasks: dict[Future[None], Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        for audio_file in audio_files:
            output_file = output_folder / audio_file.name
            if output_file.exists():
                continue
            future = executor.submit(apply_tempo, audio_file, output_file, tempo, quality)
            tasks[future] = audio_file

        if not tasks:
            LOGGER.info("全部檔案已存在，略過：%s", output_folder)
            return

        failures: list[str] = []
        with tqdm(total=len(tasks), desc=f"語速 x{tempo:g}") as progress:
            for future in as_completed(tasks):
                source = tasks[future]
                try:
                    future.result()
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append(str(source))
                    progress.write(f"處理失敗：{source}：{error}")
                progress.update(1)
    if failures:
        raise RuntimeError(f"共有 {len(failures)} 個檔案處理失敗：{', '.join(failures)}")
    LOGGER.info("語速調整完成（x%g）：%s", tempo, output_folder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合併前批次調整編號音檔語速（FFmpeg atempo）")
    parser.add_argument("--input", type=Path, required=True, help="輸入音訊資料夾")
    parser.add_argument("--output", type=Path, required=True, help="輸出資料夾")
    parser.add_argument(
        "--tempo",
        type=float,
        required=True,
        help="語速倍數，例如 0.9=變慢、1.1=變快（音高不變）",
    )
    parser.add_argument("--suffix", default=None, help="只處理指定副檔名，例如 mp3 或 wav")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="處理資料夾內所有支援音檔，不只編號檔（0.mp3、1.mp3…）",
    )
    parser.add_argument("--workers", type=int, default=None, help="並行工作數")
    parser.add_argument("--quality", type=int, default=4, help="輸出 MP3 時的 libmp3lame 品質 0-9")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process_tempo_folder(
        input_folder=args.input,
        output_folder=args.output,
        tempo=args.tempo,
        suffix=args.suffix,
        numbered_only=not args.all_files,
        max_workers=args.workers,
        quality=args.quality,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
