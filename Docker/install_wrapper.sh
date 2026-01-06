#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

cd .. || exit 1

set -e

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base

mkdir -p GPT_SoVITS/text

bash install.sh --device "CU${CUDA_VERSION//./}" --source HF

# Download FunASR models using ModelScope SDK (complete downloads)
echo "[INFO] Downloading FunASR models via ModelScope SDK..."
pip install modelscope -q

python3 -c "
from modelscope import snapshot_download
import os

models_dir = 'tools/asr/models'
os.makedirs(models_dir, exist_ok=True)

models = [
    'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', 
    'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
]

for model_id in models:
    model_name = model_id.split('/')[-1]
    local_dir = os.path.join(models_dir, model_name)
    print(f'[INFO] Downloading {model_name}...')
    snapshot_download(model_id, local_dir=local_dir)
    print(f'[INFO] Downloaded {model_name}')

print('[INFO] All FunASR models downloaded successfully!')
"

pip cache purge

pip show torch

rm -rf /tmp/* /var/tmp/*

rm -rf "$HOME/miniconda3/pkgs"

mkdir -p "$HOME/miniconda3/pkgs"

rm -rf /root/.conda /root/.cache
