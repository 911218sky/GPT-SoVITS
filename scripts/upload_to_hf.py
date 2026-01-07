#!/usr/bin/env python3
"""
上傳檔案到 Hugging Face
用於 GitHub Actions workflow: build_windows_packages.yaml

Usage:
    python scripts/upload_to_hf.py <file_path> <repo_id>

Environment:
    HF_TOKEN: Hugging Face API token (required)
"""

import os
import sys
from huggingface_hub import HfApi


def main():
    if len(sys.argv) < 3:
        print("Usage: python upload_to_hf.py <file_path> <repo_id>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    repo_id = sys.argv[2]
    token = os.environ.get('HF_TOKEN')
    
    if not token:
        print("[ERROR] HF_TOKEN environment variable is required")
        sys.exit(1)
    
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)
    
    file_name = os.path.basename(file_path)
    print(f"[INFO] Uploading {file_name} to {repo_id}...")
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        token=token
    )
    
    print(f"[INFO] Upload completed: {file_name}")


if __name__ == '__main__':
    main()
