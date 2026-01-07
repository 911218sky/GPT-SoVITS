#!/usr/bin/env python3
"""
FunASR 模型下載腳本
優先從 HuggingFace cache 下載，失敗則從 ModelScope 下載
"""

import os
import sys
import shutil

# 模型定義: (model_name, modelscope_id)
MODELS = [
    ('speech_fsmn_vad_zh-cn-16k-common-pytorch', 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'),
    ('punc_ct-transformer_zh-cn-common-vocab272727-pytorch', 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'),
    ('speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'),
]

HF_CACHE_REPO = 'sky1218/GPT-SoVITS-Models'
MODELS_DIR = 'tools/asr/models'


def download_from_hf(model_name: str, local_dir: str) -> bool:
    """從 HuggingFace cache 下載 (全球 CDN)"""
    try:
        from huggingface_hub import snapshot_download
        print(f'[INFO] Trying HuggingFace cache for {model_name}...')
        
        tmp_dir = f'{local_dir}_tmp'
        snapshot_download(
            repo_id=HF_CACHE_REPO,
            allow_patterns=f'{model_name}/*',
            local_dir=tmp_dir
        )
        
        # 從子資料夾移動到目標位置
        src = os.path.join(tmp_dir, model_name)
        if os.path.exists(src) and os.listdir(src):
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            shutil.move(src, local_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f'[INFO] Downloaded {model_name} from HuggingFace cache')
            return True
    except Exception as e:
        print(f'[INFO] HuggingFace cache not available: {e}')
    return False


def download_from_modelscope(model_name: str, modelscope_id: str, local_dir: str) -> bool:
    """從 ModelScope 下載 (主要來源)"""
    try:
        from modelscope import snapshot_download
        print(f'[INFO] Downloading {model_name} from ModelScope...')
        snapshot_download(modelscope_id, local_dir=local_dir)
        print(f'[INFO] Downloaded {model_name} from ModelScope')
        return True
    except Exception as e:
        print(f'[ERROR] ModelScope download failed: {e}')
    return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    for model_name, modelscope_id in MODELS:
        local_dir = os.path.join(MODELS_DIR, model_name)
        
        # 檢查是否已存在
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f'[INFO] {model_name} already exists, skipping')
            continue
        
        # 優先 HuggingFace，失敗則用 ModelScope
        if not download_from_hf(model_name, local_dir):
            if not download_from_modelscope(model_name, modelscope_id, local_dir):
                print(f'[ERROR] Failed to download {model_name}')
                sys.exit(1)
    
    print('[INFO] All FunASR models downloaded successfully!')


if __name__ == '__main__':
    main()
