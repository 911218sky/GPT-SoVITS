#!/usr/bin/env python3
"""
將 FunASR 模型從 ModelScope 下載並上傳到 Hugging Face
用於 GitHub Actions workflow: cache_funasr_models.yaml
支援並行下載加速
"""

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from modelscope import snapshot_download
from huggingface_hub import HfApi, create_repo

# FunASR 模型列表
MODELS = [
    'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
    'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
]

# 保留的檔案類型
ESSENTIAL_EXTENSIONS = {'.pt', '.bin', '.json', '.yaml', '.onnx', '.pth', '.safetensors'}

MAX_RETRIES = 5
MAX_WORKERS = 3  # 並行下載數量

def clean_model_dir(local_dir: str) -> None:
    """移除非必要的檔案（ModelScope metadata 等）"""
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            filepath = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext not in ESSENTIAL_EXTENSIONS:
                try:
                    os.remove(filepath)
                    print(f'[INFO] Removed {f}')
                except Exception:
                    pass
    
    # 移除空目錄
    for root, dirs, files in os.walk(local_dir, topdown=False):
        for d in dirs:
            dirpath = os.path.join(root, d)
            if not os.listdir(dirpath):
                os.rmdir(dirpath)


def download_model(model_id: str) -> tuple[str, str]:
    """下載單個模型，返回 (model_name, local_dir)"""
    model_name = model_id.split('/')[-1]
    local_dir = f'./models/{model_name}'
    
    print(f'[INFO] Downloading {model_name} from ModelScope...')
    snapshot_download(model_id, local_dir=local_dir)
    print(f'[INFO] Downloaded {model_name}')
    
    # 清理非必要檔案
    clean_model_dir(local_dir)
    
    return model_name, local_dir


def upload_to_hf(api: HfApi, local_dir: str, model_name: str, 
                 hf_repo: str, hf_token: str) -> None:
    """上傳模型到 Hugging Face（含重試邏輯）"""
    # 先刪除舊的資料夾
    try:
        api.delete_folder(
            path_in_repo=model_name,
            repo_id=hf_repo,
            token=hf_token,
            repo_type="model"
        )
        print(f'[INFO] Deleted old {model_name} folder from HF')
    except Exception as e:
        print(f'[INFO] No existing folder to delete or error: {e}')
    
    # 上傳（含重試）
    for attempt in range(MAX_RETRIES):
        try:
            print(f'[INFO] Uploading to HF: {model_name} (attempt {attempt+1}/{MAX_RETRIES})')
            api.upload_folder(
                folder_path=local_dir,
                path_in_repo=model_name,
                repo_id=hf_repo,
                token=hf_token
            )
            print(f'[INFO] Uploaded {model_name} to HF')
            return
        except Exception as e:
            print(f'[WARN] Upload failed: {e}')
            if attempt < MAX_RETRIES - 1:
                wait_time = 30 * (attempt + 1)
                print(f'[INFO] Retrying in {wait_time} seconds...')
                time.sleep(wait_time)
            else:
                raise e


def main():
    hf_repo = os.environ.get('HF_MODELS_REPO', 'sky1218/GPT-SoVITS-Models')
    hf_token = os.environ.get('HF_TOKEN')
    
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    
    # 建立 HF repo（如果不存在）
    api = HfApi()
    try:
        create_repo(repo_id=hf_repo, token=hf_token, repo_type="model", exist_ok=True)
        print(f"[INFO] Repo {hf_repo} ready")
    except Exception as e:
        print(f"[WARN] Repo creation: {e}")
    
    # 並行下載所有模型
    print(f'[INFO] Starting parallel download with {MAX_WORKERS} workers...')
    downloaded_models = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_model, model_id): model_id for model_id in MODELS}
        
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                model_name, local_dir = future.result()
                downloaded_models.append((model_name, local_dir))
            except Exception as e:
                print(f'[ERROR] Failed to download {model_id}: {e}')
                raise e
    
    print(f'[INFO] All {len(downloaded_models)} models downloaded!')
    
    # 依序上傳到 HF（避免 API 限制）
    print('[INFO] Uploading all models to Hugging Face...')
    for model_name, local_dir in downloaded_models:
        upload_to_hf(api, local_dir, model_name, hf_repo, hf_token)
        
        # 清理本地檔案
        shutil.rmtree(local_dir)
    
    print('[SUCCESS] All FunASR models cached to Hugging Face!')


if __name__ == '__main__':
    main()
