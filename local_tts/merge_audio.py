from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import subprocess
import uuid
from concurrent.futures import Future
from pathlib import Path
from typing import Final

NUMBERED_STEM = re.compile(r"^(\d+)$")
BYTES_PER_MEGABYTE: Final = 1024 * 1024


def get_sorted_audio_files(folder: Path, suffix: str) -> list[Path]:
    """只取得檔名為數字的音檔，並依數字排序。"""
    normalized_suffix = suffix.lstrip(".").lower()
    numbered_files: list[tuple[int, Path]] = []
    for path in folder.glob(f"*.{normalized_suffix}"):
        if not path.is_file():
            continue
        match = NUMBERED_STEM.fullmatch(path.stem)
        if match:
            numbered_files.append((int(match.group(1)), path))
    return [path for _, path in sorted(numbered_files, key=lambda item: (item[0], item[1].name))]


def split_files_by_size(files: list[Path], max_size: int | None) -> list[list[Path]]:
    """依檔案總大小分組，不產生空群組。"""
    if max_size is not None and max_size <= 0:
        raise ValueError("max_size 必須大於 0")
    groups: list[list[Path]] = []
    current_group: list[Path] = []
    current_size = 0
    for path in files:
        file_size = path.stat().st_size
        if current_group and max_size and current_size + file_size > max_size:
            groups.append(current_group)
            current_group = []
            current_size = 0
        current_group.append(path)
        current_size += file_size
    if current_group:
        groups.append(current_group)
    return groups


def escape_concat_path(path: Path) -> str:
    """轉義 FFmpeg concat demuxer 的單引號路徑。"""
    return str(path.resolve()).replace("'", "'\\''")


def concatenate_audio_files(audio_files: list[Path], output_path: Path) -> None:
    """使用 FFmpeg concat demuxer 合併音檔。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.parent / f".concat-{uuid.uuid4().hex}.txt"
    try:
        list_path.write_text(
            "".join(f"file '{escape_concat_path(path)}'\n" for path in audio_files),
            encoding="utf-8",
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() or "沒有 FFmpeg 錯誤訊息"
            raise RuntimeError(f"FFmpeg 合併失敗：{output_path}\n{detail}") from error
    finally:
        list_path.unlink(missing_ok=True)


def merge_audios(
    input_folder: Path,
    output_dir: Path,
    output_file_name: str = "tts_powerful_output.mp3",
    max_total_size: int | None = 1 * 1024 * 1024 * 1024,
    audio_suffix: str = "mp3",
    max_workers: int | None = None,
) -> list[Path]:
    """依編號排序並分組合併音檔。"""
    if not input_folder.is_dir():
        raise NotADirectoryError(f"輸入資料夾不存在：{input_folder}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，請先安裝並加入 PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = get_sorted_audio_files(input_folder, audio_suffix)
    if not audio_files:
        raise FileNotFoundError(f"找不到編號音檔：{input_folder}/*.{audio_suffix.lstrip('.')}")

    merge_groups = split_files_by_size(audio_files, max_total_size)
    worker_count = max_workers or min(len(merge_groups), 4)
    results: list[Path] = []
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        futures: dict[Future[None], Path] = {}
        for index, group in enumerate(merge_groups, start=1):
            output_path = output_dir / f"{index}_{output_file_name}"
            if output_path.exists():
                results.append(output_path)
                continue
            futures[executor.submit(concatenate_audio_files, group, output_path)] = output_path
        for future in concurrent.futures.as_completed(futures):
            output_path = futures[future]
            try:
                future.result()
                results.append(output_path)
            except (OSError, RuntimeError) as error:
                failures.append(str(output_path))
                print(f"合併處理失敗：{error}")
    if failures:
        raise RuntimeError(f"共有 {len(failures)} 組合併失敗：{', '.join(failures)}")
    return sorted(results)


def parse_args() -> argparse.Namespace:
    def positive_megabytes(value: str) -> int:
        megabytes = int(value)
        if megabytes <= 0:
            raise argparse.ArgumentTypeError("必須是大於 0 的整數 MB")
        return megabytes

    parser = argparse.ArgumentParser(description="依編號合併多個音訊檔")
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="tts_powerful_output.mp3")
    parser.add_argument(
        "--max-size-mb",
        type=positive_megabytes,
        default=1024,
        help="每個合併檔的來源音檔總大小上限，單位為 MB（預設：1024）",
    )
    parser.add_argument("--suffix", default="mp3")
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """解析 CLI 並執行音檔合併。"""
    args = parse_args()
    outputs = merge_audios(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        output_file_name=args.output_name,
        max_total_size=args.max_size_mb * BYTES_PER_MEGABYTE,
        audio_suffix=args.suffix,
        max_workers=args.workers,
    )
    print(f"完成 {len(outputs)} 個合併檔：{args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
