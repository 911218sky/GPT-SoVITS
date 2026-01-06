#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

cd .. || exit 1

set -e

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

mkdir -p GPT_SoVITS/text

bash install.sh --device "CU${CUDA_VERSION//./}" --source HF

# Download FunASR models (try HuggingFace cache first, fallback to ModelScope)
echo "[INFO] Downloading FunASR models..."
pip install huggingface_hub modelscope -q

python3 -c "
import os
import sys

models_dir = 'tools/asr/models'
os.makedirs(models_dir, exist_ok=True)

# Model definitions: (model_name, modelscope_id)
# HF cache path will be: sky1218/GPT-SoVITS-Models/{model_name}/
models = [
    ('speech_fsmn_vad_zh-cn-16k-common-pytorch', 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'),
    ('punc_ct-transformer_zh-cn-common-vocab272727-pytorch', 'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'),
    ('speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'),
]

HF_CACHE_REPO = 'sky1218/GPT-SoVITS-Models'

def download_from_hf(model_name, local_dir):
    \"\"\"Try downloading from user's HuggingFace cache (global CDN)\"\"\"
    try:
        from huggingface_hub import snapshot_download
        print(f'[INFO] Trying HuggingFace cache for {model_name}...')
        snapshot_download(
            repo_id=HF_CACHE_REPO,
            allow_patterns=f'{model_name}/*',
            local_dir=local_dir + '_tmp'
        )
        # Move from subfolder to target
        import shutil
        src = os.path.join(local_dir + '_tmp', model_name)
        if os.path.exists(src) and os.listdir(src):
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            shutil.move(src, local_dir)
            shutil.rmtree(local_dir + '_tmp', ignore_errors=True)
            print(f'[INFO] Downloaded {model_name} from HuggingFace cache')
            return True
    except Exception as e:
        print(f'[INFO] HuggingFace cache not available: {e}')
    return False

def download_from_modelscope(model_name, modelscope_id, local_dir):
    \"\"\"Download from ModelScope (primary source)\"\"\"
    try:
        from modelscope import snapshot_download
        print(f'[INFO] Downloading {model_name} from ModelScope...')
        snapshot_download(modelscope_id, local_dir=local_dir)
        print(f'[INFO] Downloaded {model_name} from ModelScope')
        return True
    except Exception as e:
        print(f'[ERROR] ModelScope download failed: {e}')
    return False

for model_name, modelscope_id in models:
    local_dir = os.path.join(models_dir, model_name)
    
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f'[INFO] {model_name} already exists, skipping')
        continue
    
    # Try HuggingFace cache first, then ModelScope
    if not download_from_hf(model_name, local_dir):
        if not download_from_modelscope(model_name, modelscope_id, local_dir):
            print(f'[ERROR] Failed to download {model_name}')
            sys.exit(1)

print('[INFO] All FunASR models downloaded successfully!')
"

pip cache purge

pip show torch

rm -rf /tmp/* /var/tmp/*

rm -rf "$HOME/miniconda3/pkgs"

mkdir -p "$HOME/miniconda3/pkgs"

rm -rf /root/.conda /root/.cache
