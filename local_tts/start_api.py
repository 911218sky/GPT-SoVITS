from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

from .common import PROJECT_ROOT, ROLE_PROFILES, get_role_profile, require_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="啟動 GPT-SoVITS api_v2.py")
    parser.add_argument("--host", default=os.environ.get("GPT_SOVITS_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GPT_SOVITS_API_PORT", "9880")))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml",
    )
    parser.add_argument("--role", choices=sorted(ROLE_PROFILES), default=None)
    parser.add_argument("--gpt-weights", type=Path, default=None)
    parser.add_argument("--sovits-weights", type=Path, default=None)
    parser.add_argument("--wait-timeout", type=float, default=900.0)
    parser.add_argument("--no-set-model", action="store_true")
    return parser.parse_args()


def wait_until_ready(process: subprocess.Popen[bytes], base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = f"{base_url}/docs"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API 程序提前結束，返回碼：{process.returncode}")
        try:
            response = requests.get(health_url, timeout=3)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"API 在 {timeout:.0f} 秒內沒有準備完成：{health_url}")


def set_model(base_url: str, gpt_weights: Path, sovits_weights: Path) -> None:
    for endpoint, weights_path in (
        ("/set_gpt_weights", gpt_weights),
        ("/set_sovits_weights", sovits_weights),
    ):
        response = requests.get(
            f"{base_url}{endpoint}",
            params={"weights_path": str(require_file(weights_path, "模型權重"))},
            timeout=900,
        )
        response.raise_for_status()
        print(f"{endpoint} 已設置：{weights_path}", flush=True)


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"TTS 設定檔不存在：{args.config}")

    gpt_weights = args.gpt_weights
    sovits_weights = args.sovits_weights
    if args.role:
        profile = get_role_profile(args.role)
        gpt_weights = gpt_weights or profile.gpt_weights_path
        sovits_weights = sovits_weights or profile.sovits_weights_path
    if (gpt_weights is None) != (sovits_weights is None):
        raise ValueError("--gpt-weights 與 --sovits-weights 必須同時指定")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "api_v2.py"),
        "--bind_addr",
        args.host,
        "--port",
        str(args.port),
        "--tts_config",
        str(args.config),
    ]
    print(f"啟動 API：{' '.join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    base_url = f"http://{'127.0.0.1' if args.host in {'0.0.0.0', '::'} else args.host}:{args.port}"
    try:
        wait_until_ready(process, base_url, args.wait_timeout)
        print(f"API 已就緒：{base_url}", flush=True)
        if not args.no_set_model and gpt_weights and sovits_weights:
            set_model(base_url, gpt_weights, sovits_weights)
        return process.wait()
    except KeyboardInterrupt:
        return 130
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            except KeyboardInterrupt:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
