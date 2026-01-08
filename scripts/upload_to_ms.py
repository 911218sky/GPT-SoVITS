#!/usr/bin/env python3
"""Upload file to ModelScope repository."""

import os
import sys
import shutil
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: python upload_to_ms.py <file_path> <repo_id>")
        print("Example: python upload_to_ms.py package.tar.zst sky1218/GPT-SoVITS-Packages")
        sys.exit(1)

    file_path = sys.argv[1]
    repo_id = sys.argv[2]

    ms_token = os.environ.get("MS_TOKEN")
    if not ms_token:
        print("[ERROR] MS_TOKEN environment variable not set")
        sys.exit(1)

    if not Path(file_path).exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    try:
        from modelscope.hub.api import HubApi

        api = HubApi()
        api.login(ms_token)

        # Create temp folder for upload
        upload_dir = Path("ms_upload")
        upload_dir.mkdir(exist_ok=True)
        shutil.copy(file_path, upload_dir)

        print(f"[INFO] Uploading {file_path} to ModelScope repo: {repo_id}")

        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(upload_dir),
            revision="master",
            commit_message=f"Upload {Path(file_path).name}",
        )

        print(f"[INFO] Successfully uploaded {file_path} to ModelScope")

        # Cleanup
        shutil.rmtree(upload_dir, ignore_errors=True)

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
