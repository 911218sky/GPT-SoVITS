from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".aac", ".flac", ".m4a"})


@dataclass(frozen=True, slots=True)
class ProcessStep:
    volume_boost: float
    silence_duration: float
    silence_threshold: int
    quality: int
    max_workers: int | None = None


def remove_silence(
    input_file: Path,
    output_file: Path,
    silence_duration: float = 0.3,
    silence_threshold: int = -30,
    quality: int = 4,
    volume_boost: float = 1.0,
) -> None:
    """使用 FFmpeg 移除靜音並輸出 MP3。"""
    if silence_duration < 0:
        raise ValueError("silence_duration 必須大於等於 0")
    if quality < 0 or quality > 9:
        raise ValueError("quality 必須介於 0 到 9")
    if volume_boost <= 0:
        raise ValueError("volume_boost 必須大於 0")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "silenceremove=stop_periods=-1:"
        f"stop_duration={silence_duration}:"
        f"stop_threshold={silence_threshold}dB,"
        f"volume={volume_boost}"
    )
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
        "-c:a",
        "libmp3lame",
        "-q:a",
        str(quality),
        str(output_file),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or "沒有 FFmpeg 錯誤訊息"
        raise RuntimeError(f"FFmpeg 處理失敗：{input_file}\n{detail}") from error


def process_files(
    input_folder: Path,
    output_folder: Path,
    max_workers: int | None = None,
    volume_boost: float = 1.0,
    silence_duration: float = 0.3,
    silence_threshold: int = -30,
    quality: int = 4,
) -> None:
    """並行處理資料夾內的音訊檔案。"""
    if not input_folder.is_dir():
        raise NotADirectoryError(f"輸入資料夾不存在：{input_folder}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，請先安裝並加入 PATH")

    output_folder.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(
        (path for path in input_folder.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: path.name,
    )
    if not audio_files:
        LOGGER.warning("在 %s 中找不到支援的音訊檔", input_folder)
        return

    worker_count = max_workers or max(1, (os.cpu_count() or 4) - 2)
    tasks: dict[Future[None], Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        for audio_file in audio_files:
            output_file = output_folder / f"{audio_file.stem}.mp3"
            if output_file.exists():
                continue
            future: Future[None] = executor.submit(
                remove_silence,
                audio_file,
                output_file,
                silence_duration,
                silence_threshold,
                quality,
                volume_boost,
            )
            tasks[future] = audio_file

        failures: list[str] = []
        with tqdm(total=len(tasks), desc="處理音訊檔案") as progress:
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


def move_filtered_files(source_folder: Path, destination_folder: Path, pattern: str = "*.mp3") -> None:
    """將來源資料夾的檔案搬到目的資料夾，不處理子資料夾。"""
    if not source_folder.is_dir():
        raise NotADirectoryError(f"來源資料夾不存在：{source_folder}")
    destination_folder.mkdir(parents=True, exist_ok=True)
    for file_path in source_folder.glob(pattern):
        if file_path.is_file():
            shutil.move(str(file_path), str(destination_folder / file_path.name))


def run_pipeline(input_folder: Path, final_output_folder: Path, steps: list[ProcessStep]) -> None:
    """依序執行去靜音步驟，最後只保留 MP3 結果。"""
    if not steps:
        raise ValueError("至少要提供一個音訊處理步驟")
    temporary_folder = final_output_folder / "_tmp_steps"
    current_input = input_folder
    try:
        for index, parameters in enumerate(steps, start=1):
            step_output = temporary_folder / f"step_{index}"
            if step_output.exists():
                shutil.rmtree(step_output)
            process_files(
                input_folder=current_input,
                output_folder=step_output,
                max_workers=parameters.max_workers,
                volume_boost=parameters.volume_boost,
                silence_duration=parameters.silence_duration,
                silence_threshold=parameters.silence_threshold,
                quality=parameters.quality,
            )
            current_input = step_output
        move_filtered_files(current_input, final_output_folder)
    finally:
        if temporary_folder.exists():
            shutil.rmtree(temporary_folder)
    LOGGER.info("所有步驟完成，結果位於：%s", final_output_folder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批次移除音訊靜音並調整音量")
    parser.add_argument("--input", type=Path, required=True, help="輸入音訊資料夾")
    parser.add_argument("--output", type=Path, required=True, help="最終輸出資料夾")
    parser.add_argument("--workers", type=int, default=None, help="並行工作數")
    parser.add_argument("--volume-boost", type=float, default=1.0)
    parser.add_argument("--silence-duration", type=float, default=0.5)
    parser.add_argument("--silence-threshold", type=int, default=-30)
    parser.add_argument("--quality", type=int, default=4)
    parser.add_argument("--single-step", action="store_true", help="只執行一次去靜音")
    return parser.parse_args()


def main() -> int:
    """解析 CLI 並執行音訊清理流程。"""
    args = parse_args()
    if args.single_step:
        steps = [
            ProcessStep(
                volume_boost=args.volume_boost,
                silence_duration=args.silence_duration,
                silence_threshold=args.silence_threshold,
                quality=args.quality,
                max_workers=args.workers,
            )
        ]
    else:
        steps = [
            ProcessStep(args.volume_boost, args.silence_duration, args.silence_threshold, args.quality, args.workers),
            ProcessStep(args.volume_boost, 2.0, -20, args.quality, args.workers),
        ]
    run_pipeline(args.input, args.output, steps)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
