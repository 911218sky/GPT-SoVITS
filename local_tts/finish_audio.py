from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import shutil
import subprocess
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".aac", ".flac", ".m4a"})
NUMBERED_STEM = re.compile(r"^(\d+)$")
BYTES_PER_MEGABYTE: Final = 1024 * 1024


def default_workers() -> int:
    """未指定 --workers 時用滿全部邏輯核心。"""
    return max(1, os.cpu_count() or 4)


# ---------------------------------------------------------------------------
# 清靜音
# ---------------------------------------------------------------------------


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
    _require_ffmpeg()

    output_folder.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(
        (path for path in input_folder.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: path.name,
    )
    if not audio_files:
        LOGGER.warning("在 %s 中找不到支援的音訊檔", input_folder)
        return

    worker_count = max_workers or default_workers()
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
    LOGGER.info("清靜音完成：%s", final_output_folder)


def clean_steps_from_args(
    *,
    volume_boost: float,
    silence_steps: list[tuple[float, int]],
    quality: int,
    workers: int | None,
) -> list[ProcessStep]:
    if not silence_steps:
        raise ValueError("至少要一輪靜音步驟")
    return [
        ProcessStep(volume_boost, duration, threshold, quality, workers)
        for duration, threshold in silence_steps
    ]


# ---------------------------------------------------------------------------
# 調速
# ---------------------------------------------------------------------------


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
    _require_ffmpeg()

    audio_files = list_audio_files(input_folder, suffix, numbered_only)
    if not audio_files:
        LOGGER.warning("在 %s 中找不到可處理的音訊檔", input_folder)
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    worker_count = max_workers or default_workers()
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


# ---------------------------------------------------------------------------
# 合併
# ---------------------------------------------------------------------------


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
    _require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = get_sorted_audio_files(input_folder, audio_suffix)
    if not audio_files:
        raise FileNotFoundError(f"找不到編號音檔：{input_folder}/*.{audio_suffix.lstrip('.')}")

    merge_groups = split_files_by_size(audio_files, max_total_size)
    worker_count = max_workers or min(len(merge_groups), default_workers())
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
                LOGGER.error("合併處理失敗：%s", error)
    if failures:
        raise RuntimeError(f"共有 {len(failures)} 組合併失敗：{', '.join(failures)}")
    return sorted(results)


# ---------------------------------------------------------------------------
# 加速路徑：一段 WAV 只編碼一次（多輪靜音鏈 + 可選調速），再 copy 合併
# ---------------------------------------------------------------------------

DEFAULT_SILENCE_STEPS = "0.5:-30,2.0:-20"


def parse_silence_steps(value: str) -> list[tuple[float, int]]:
    """解析多輪靜音：`秒數:閾值dB`，逗號或分號分隔。例：`0.5:-30,2.0:-20`。"""
    text = (value or "").strip()
    if not text:
        raise argparse.ArgumentTypeError("靜音步驟不可為空，例如 0.5:-30,2.0:-20")

    steps: list[tuple[float, int]] = []
    for raw in re.split(r"[,;]", text):
        part = raw.strip()
        if not part:
            continue
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"靜音步驟格式錯誤「{part}」，應為 秒數:閾值dB，例如 0.5:-30"
            )
        duration_s, threshold_s = part.split(":", 1)
        try:
            duration = float(duration_s.strip())
            threshold = int(float(threshold_s.strip()))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"靜音步驟無法解析「{part}」，應為 秒數:閾值dB"
            ) from error
        if duration < 0:
            raise argparse.ArgumentTypeError(f"靜音秒數不可為負：{duration}")
        steps.append((duration, threshold))

    if not steps:
        raise argparse.ArgumentTypeError("至少要一輪靜音步驟，例如 0.5:-30")
    return steps


def build_prepare_filter(
    *,
    volume_boost: float,
    silence_steps: list[tuple[float, int]],
    tempo: float,
) -> str:
    """把多輪 silenceremove（每輪後 volume）+ 可選 atempo 串成一條 filter。"""
    if not silence_steps:
        raise ValueError("至少要一輪靜音步驟")
    parts: list[str] = []
    for duration, threshold in silence_steps:
        parts.append(
            "silenceremove=stop_periods=-1:"
            f"stop_duration={duration}:"
            f"stop_threshold={threshold}dB"
        )
        parts.append(f"volume={volume_boost}")
    if abs(tempo - 1.0) > 1e-6:
        parts.append(build_atempo_filter(tempo))
    return ",".join(parts)


def prepare_one(
    input_file: Path,
    output_file: Path,
    *,
    volume_boost: float,
    silence_steps: list[tuple[float, int]],
    tempo: float,
    quality: int,
) -> None:
    """單檔：清靜音（+調速）一次編碼成 MP3。"""
    if quality < 0 or quality > 9:
        raise ValueError("quality 必須介於 0 到 9")
    if volume_boost <= 0:
        raise ValueError("volume_boost 必須大於 0")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = build_prepare_filter(
        volume_boost=volume_boost,
        silence_steps=silence_steps,
        tempo=tempo,
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
        raise RuntimeError(f"FFmpeg 預處理失敗：{input_file}\n{detail}") from error


def prepare_folder(
    input_folder: Path,
    output_folder: Path,
    *,
    volume_boost: float = 1.0,
    silence_steps: list[tuple[float, int]] | None = None,
    tempo: float = 1.0,
    quality: int = 4,
    max_workers: int | None = None,
) -> None:
    """並行：每段只壓一次 MP3（效果對齊 clean±tempo）。"""
    if not input_folder.is_dir():
        raise NotADirectoryError(f"輸入資料夾不存在：{input_folder}")
    _require_ffmpeg()
    steps = silence_steps or parse_silence_steps(DEFAULT_SILENCE_STEPS)

    audio_files = list_audio_files(input_folder, suffix=None, numbered_only=True)
    if not audio_files:
        # 允許非編號（少見）；退回所有支援副檔名
        audio_files = sorted(
            (p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS),
            key=lambda p: p.name,
        )
    if not audio_files:
        raise FileNotFoundError(f"找不到可處理音檔：{input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    worker_count = max_workers or default_workers()
    tasks: dict[Future[None], Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
        for audio_file in audio_files:
            output_file = output_folder / f"{audio_file.stem}.mp3"
            if output_file.exists():
                continue
            tasks[
                executor.submit(
                    prepare_one,
                    audio_file,
                    output_file,
                    volume_boost=volume_boost,
                    silence_steps=steps,
                    tempo=tempo,
                    quality=quality,
                )
            ] = audio_file

        if not tasks:
            LOGGER.info("預處理檔案皆已存在：%s", output_folder)
            return

        desc = f"清音×{len(steps)}" + ("+調速" if abs(tempo - 1.0) > 1e-6 else "")
        failures: list[str] = []
        with tqdm(total=len(tasks), desc=desc) as progress:
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
    LOGGER.info("預處理完成：%s", output_folder)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("找不到 ffmpeg，請先安裝並加入 PATH")


def _positive_megabytes(value: str) -> int:
    megabytes = int(value)
    if megabytes <= 0:
        raise argparse.ArgumentTypeError("必須是大於 0 的整數 MB")
    return megabytes


def _add_clean_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"並行工作數（預設用滿全部 CPU，目前 {default_workers()}）",
    )
    parser.add_argument("--volume-boost", type=float, default=1.0)
    parser.add_argument(
        "--silence-steps",
        type=parse_silence_steps,
        default=parse_silence_steps(DEFAULT_SILENCE_STEPS),
        help=f"多輪靜音 list：秒數:閾值dB，逗號分隔（預設 {DEFAULT_SILENCE_STEPS}）",
    )
    parser.add_argument("--quality", type=int, default=4, help="MP3 品質 0-9（越大越快檔越小）")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="音訊後處理：clean / tempo / merge / all（推薦 all＝一次編碼再合併）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser(
        "all",
        help="推薦：每段只編碼一次（清音±調速）後合併，效果同三步串接但更快",
    )
    p_all.add_argument("--input", type=Path, required=True, help="batch_tts 編號 WAV 目錄")
    p_all.add_argument("--output-dir", type=Path, default=None, help="合併成品目錄（預設 <input>_merged）")
    p_all.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="中繼 MP3 目錄（預設 <input>_prepared）",
    )
    p_all.add_argument("--tempo", type=float, default=1.0, help="語速，1.0=不調")
    p_all.add_argument("--max-size-mb", type=_positive_megabytes, default=1024)
    p_all.add_argument("--output-name", default="tts_powerful_output.mp3")
    p_all.add_argument(
        "--legacy",
        action="store_true",
        help="改走舊三步（多輪清音各編碼一次 + 調速再編碼 + 合併），較慢",
    )
    _add_clean_flags(p_all)

    p_clean = sub.add_parser("clean", help="只清靜音（每輪各編碼一次；要快請用 all）")
    p_clean.add_argument("--input", type=Path, required=True)
    p_clean.add_argument("--output", type=Path, required=True)
    _add_clean_flags(p_clean)

    p_tempo = sub.add_parser("tempo", help="只調語速")
    p_tempo.add_argument("--input", type=Path, required=True)
    p_tempo.add_argument("--output", type=Path, required=True)
    p_tempo.add_argument("--tempo", type=float, required=True)
    p_tempo.add_argument("--suffix", default=None)
    p_tempo.add_argument("--all-files", action="store_true")
    p_tempo.add_argument("--workers", type=int, default=None, help=f"並行數（預設用滿 CPU={default_workers()}）")
    p_tempo.add_argument("--quality", type=int, default=4)

    p_merge = sub.add_parser("merge", help="只合併編號音檔")
    p_merge.add_argument("--input-folder", "--input", type=Path, required=True, dest="input_folder")
    p_merge.add_argument("--output-dir", "--output", type=Path, required=True, dest="output_dir")
    p_merge.add_argument("--output-name", default="tts_powerful_output.mp3")
    p_merge.add_argument("--max-size-mb", type=_positive_megabytes, default=1024)
    p_merge.add_argument("--suffix", default="mp3")
    p_merge.add_argument("--workers", type=int, default=None, help=f"並行數（預設用滿 CPU={default_workers()}）")

    return parser.parse_args(argv)


def cmd_clean(args: argparse.Namespace) -> int:
    steps = clean_steps_from_args(
        volume_boost=args.volume_boost,
        silence_steps=args.silence_steps,
        quality=args.quality,
        workers=args.workers,
    )
    LOGGER.info("清靜音輪次：%s", ",".join(f"{d}:{t}" for d, t in args.silence_steps))
    run_pipeline(args.input, args.output, steps)
    return 0


def cmd_tempo(args: argparse.Namespace) -> int:
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


def cmd_merge(args: argparse.Namespace) -> int:
    outputs = merge_audios(
        input_folder=args.input_folder,
        output_dir=args.output_dir,
        output_file_name=args.output_name,
        max_total_size=args.max_size_mb * BYTES_PER_MEGABYTE,
        audio_suffix=args.suffix,
        max_workers=args.workers,
    )
    LOGGER.info("完成 %d 個合併檔：%s", len(outputs), args.output_dir)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"輸入資料夾不存在：{input_dir}")

    work_dir = (args.work_dir or Path(f"{input_dir}_prepared")).resolve()
    output_dir = (args.output_dir or Path(f"{input_dir}_merged")).resolve()
    do_tempo = abs(args.tempo - 1.0) > 1e-6

    LOGGER.info("=== finish_audio all ===")
    LOGGER.info("input=%s", input_dir)
    LOGGER.info(
        "work=%s  merged=%s  tempo=%g  legacy=%s  silence_steps=%s",
        work_dir,
        output_dir,
        args.tempo,
        args.legacy,
        ",".join(f"{d}:{t}" for d, t in args.silence_steps),
    )

    if args.legacy:
        clean_dir = Path(f"{input_dir}_clean")
        LOGGER.info("[legacy 1/3] 清靜音 → %s", clean_dir)
        run_pipeline(
            input_dir,
            clean_dir,
            clean_steps_from_args(
                volume_boost=args.volume_boost,
                silence_steps=args.silence_steps,
                quality=args.quality,
                workers=args.workers,
            ),
        )
        merge_source = clean_dir
        if do_tempo:
            tempo_dir = Path(f"{input_dir}_tempo")
            LOGGER.info("[legacy 2/3] 調速 → %s", tempo_dir)
            process_tempo_folder(
                clean_dir,
                tempo_dir,
                tempo=args.tempo,
                suffix="mp3",
                numbered_only=True,
                max_workers=args.workers,
                quality=args.quality,
            )
            merge_source = tempo_dir
        else:
            LOGGER.info("[legacy 2/3] 略過調速")
        LOGGER.info("[legacy 3/3] 合併 → %s", output_dir)
        outputs = merge_audios(
            merge_source,
            output_dir,
            output_file_name=args.output_name,
            max_total_size=args.max_size_mb * BYTES_PER_MEGABYTE,
            audio_suffix="mp3",
            max_workers=args.workers,
        )
    else:
        LOGGER.info("[1/2] 單次編碼預處理（清音%s）→ %s", "+調速" if do_tempo else "", work_dir)
        prepare_folder(
            input_dir,
            work_dir,
            volume_boost=args.volume_boost,
            silence_steps=args.silence_steps,
            tempo=args.tempo,
            quality=args.quality,
            max_workers=args.workers,
        )
        LOGGER.info("[2/2] 合併 → %s", output_dir)
        outputs = merge_audios(
            work_dir,
            output_dir,
            output_file_name=args.output_name,
            max_total_size=args.max_size_mb * BYTES_PER_MEGABYTE,
            audio_suffix="mp3",
            max_workers=args.workers,
        )

    LOGGER.info("完成 %d 個合併檔：%s", len(outputs), output_dir)
    for path in outputs:
        LOGGER.info("  %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {
        "all": cmd_all,
        "clean": cmd_clean,
        "tempo": cmd_tempo,
        "merge": cmd_merge,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
