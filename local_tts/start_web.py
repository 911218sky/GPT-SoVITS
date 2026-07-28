from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .common import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="啟動 GPT-SoVITS WebUI")
    parser.add_argument("--language", default="Auto", help="WebUI 語言，例如 zh、en、Auto")
    parser.add_argument("--host", default="127.0.0.1", help="WebUI 綁定地址")
    parser.add_argument("--port", type=int, default=None, help="WebUI 主埠號")
    parser.add_argument("--share", action="store_true", help="啟用 Gradio 公開分享連結")
    parser.add_argument("--cpu", action="store_true", help="強制使用 CPU")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    environment = os.environ.copy()
    environment["GPT_SOVITS_WEBUI_HOST"] = args.host
    if args.port is not None:
        environment["GPT_SOVITS_WEBUI_PORT"] = str(args.port)
    if args.share:
        environment["is_share"] = "True"
    if args.cpu:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["is_half"] = "False"

    command = [sys.executable, str(PROJECT_ROOT / "webui.py"), args.language]
    print(f"啟動 WebUI：{' '.join(command)}", flush=True)
    try:
        return subprocess.call(command, cwd=PROJECT_ROOT, env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
