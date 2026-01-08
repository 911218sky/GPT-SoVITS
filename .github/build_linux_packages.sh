#!/bin/bash
set -e

echo "=========================================="
echo "  GPT-SoVITS Linux Package Builder"
echo "=========================================="

# === Configuration ===
TORCH_TARGET="${TORCH_CUDA:-cu126}"
DATE_SUFFIX="${DATE_SUFFIX:-$(date +%m%d)}"
PKG_SUFFIX="${PKG_SUFFIX:-}"

PKG_NAME="GPT-SoVITS-${DATE_SUFFIX}${PKG_SUFFIX}-${TORCH_TARGET}-linux"

SRC_DIR="$(pwd)"
TMP_DIR="${SRC_DIR}/tmp"

# HuggingFace base URL
HF_BASE="https://huggingface.co/lj1995/GPT-SoVITS/resolve/main"

echo "[INFO] Package: ${PKG_NAME}"
echo "[INFO] Target: ${TORCH_TARGET}"

# === Helper Functions ===
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=15
    local delay=5
    
    for ((i=1; i<=max_retries; i++)); do
        if wget -q --show-progress --tries=3 --timeout=60 "$url" -O "$output" 2>/dev/null; then
            return 0
        fi
        echo "[INFO] Download attempt $i failed, retrying..."
        sleep $delay
    done
    echo "[WARNING] Download failed after $max_retries attempts: $url"
    return 1
}

download_hf_folder() {
    local repo_path="$1"
    local local_dir="$2"
    
    mkdir -p "$local_dir"
    local api_url="https://huggingface.co/api/models/lj1995/GPT-SoVITS/tree/main/${repo_path}"
    local files=$(curl -s "$api_url" | jq -r '.[] | select(.type=="file") | .path')
    
    for file_path in $files; do
        local file_name=$(basename "$file_path")
        local download_url="${HF_BASE}/${file_path}"
        echo "  -> ${file_name}"
        download_with_retry "$download_url" "${local_dir}/${file_name}"
    done
}

# === Cleanup ===
echo ""
echo "[1/8] Cleaning up..."
rm -rf "${SRC_DIR}/.git"
mkdir -p "$TMP_DIR"

# ============================================
# PHASE 1: Download all resources
# ============================================

echo ""
echo "=========================================="
echo "  PHASE 1: Downloading Resources"
echo "=========================================="

# === Download Pretrained Models ===
echo ""
echo "[2/8] Downloading pretrained models from HuggingFace..."
PRETRAINED_DIR="${SRC_DIR}/GPT_SoVITS/pretrained_models"
mkdir -p "$PRETRAINED_DIR"

echo "[INFO] Downloading gsv-v2final-pretrained..."
download_hf_folder "gsv-v2final-pretrained" "${PRETRAINED_DIR}/gsv-v2final-pretrained"

echo "[INFO] Downloading chinese-hubert-base..."
download_hf_folder "chinese-hubert-base" "${PRETRAINED_DIR}/chinese-hubert-base"

echo "[INFO] Downloading chinese-roberta-wwm-ext-large..."
download_hf_folder "chinese-roberta-wwm-ext-large" "${PRETRAINED_DIR}/chinese-roberta-wwm-ext-large"

echo "[INFO] Downloading v2Pro..."
download_hf_folder "v2Pro" "${PRETRAINED_DIR}/v2Pro"

echo "[INFO] Downloading sv model..."
mkdir -p "${PRETRAINED_DIR}/sv"
download_with_retry "${HF_BASE}/sv/pretrained_eres2netv2w24s4ep4.ckpt" "${PRETRAINED_DIR}/sv/pretrained_eres2netv2w24s4ep4.ckpt"

echo "[INFO] Downloading s1v3.ckpt..."
download_with_retry "${HF_BASE}/s1v3.ckpt" "${PRETRAINED_DIR}/s1v3.ckpt"

# === Download G2PW Model ===
echo ""
echo "[3/8] Downloading G2PW model..."
G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
download_with_retry "$G2PW_URL" "${TMP_DIR}/G2PWModel.zip"
unzip -q -o "${TMP_DIR}/G2PWModel.zip" -d "${SRC_DIR}/GPT_SoVITS/text"
rm -f "${TMP_DIR}/G2PWModel.zip"

# === Download fast-langdetect model ===
echo ""
echo "[4/8] Downloading fast-langdetect model..."
FAST_LANGDETECT_DIR="${PRETRAINED_DIR}/fast_langdetect"
mkdir -p "$FAST_LANGDETECT_DIR"
download_with_retry "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin" "${FAST_LANGDETECT_DIR}/lid.176.bin"

echo ""
echo "[INFO] All resources downloaded!"

# ============================================
# PHASE 2: Setup Python Environment
# ============================================

echo ""
echo "=========================================="
echo "  PHASE 2: Setting up Environment"
echo "=========================================="

# === Install uv and Create Python Environment ===
echo ""
echo "[5/8] Installing uv and setting up portable Python environment..."
RUNTIME_PATH="${SRC_DIR}/runtime"
ENV_PATH="${RUNTIME_PATH}/python"

mkdir -p "$RUNTIME_PATH"

# Download and install uv (to a separate directory to avoid conflict)
echo "[INFO] Downloading uv..."
UV_INSTALL_DIR="${RUNTIME_PATH}/uv_bin"
mkdir -p "$UV_INSTALL_DIR"
curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_INSTALL_DIR" sh

UV="${UV_INSTALL_DIR}/uv"

if [ ! -f "$UV" ]; then
    echo "[ERROR] uv not found at $UV"
    exit 1
fi

# Download standalone Python
echo "[INFO] Downloading standalone Python 3.11..."
PYTHON_VERSION="3.11.14"
PYTHON_RELEASE="20251217"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-x86_64-unknown-linux-gnu-install_only.tar.gz"

download_with_retry "$PYTHON_URL" "${TMP_DIR}/python.tar.gz"

echo "[INFO] Extracting Python..."
tar -xzf "${TMP_DIR}/python.tar.gz" -C "$TMP_DIR"

# Move python folder to runtime
if [ -d "${TMP_DIR}/python" ]; then
    mv "${TMP_DIR}/python" "$ENV_PATH"
fi
rm -f "${TMP_DIR}/python.tar.gz"

PYTHON="${ENV_PATH}/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] python not found at $PYTHON"
    exit 1
fi

echo "[INFO] Standalone Python installed at: $PYTHON"

# Set UV timeout for large downloads
export UV_HTTP_TIMEOUT=300

# === Install PyTorch ===
echo ""
echo "[6/8] Installing PyTorch (${TORCH_TARGET})..."

TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
TORCHAUDIO_VERSION="2.8.0"

case "$TORCH_TARGET" in
    cu126)
        $UV pip install --python "$PYTHON" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu126
        ;;
    cu128)
        $UV pip install --python "$PYTHON" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu128
        ;;
    cu129)
        $UV pip install --python "$PYTHON" torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu129
        ;;
    cu130)
        # CUDA 13.0 requires PyTorch 2.9.0
        $UV pip install --python "$PYTHON" torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
        ;;
    *)
        echo "[ERROR] Unsupported target: ${TORCH_TARGET} (supported: cu126, cu128, cu129, cu130)"
        exit 1
        ;;
esac

# === Install Dependencies ===
echo ""
echo "[7/8] Installing dependencies..."

# Remove --no-binary constraint for faster installation
grep -v "^--no-binary" requirements.txt > requirements_optimized.txt || cp requirements.txt requirements_optimized.txt

$UV pip install --python "$PYTHON" -r requirements_optimized.txt
$UV pip install --python "$PYTHON" -r extra-req.txt

rm -f requirements_optimized.txt

# Cleanup caches
echo "[INFO] Cleaning up caches..."
$UV cache clean

# Download NLTK Data
echo "[INFO] Downloading NLTK data..."
$PYTHON -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# Download FunASR models
echo "[INFO] Downloading FunASR models..."
$UV pip install --python "$PYTHON" "huggingface_hub[hf_xet]" -q

ASR_MODELS_DIR="${SRC_DIR}/tools/asr/models"
mkdir -p "$ASR_MODELS_DIR"

HF_CACHE_REPO="${HF_MODELS_REPO:-sky1218/GPT-SoVITS-Models}"

FUNASR_MODELS=(
    "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)

for model_name in "${FUNASR_MODELS[@]}"; do
    local_dir="${ASR_MODELS_DIR}/${model_name}"
    if [ -d "$local_dir" ] && [ "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "[INFO] ${model_name} already exists, skipping"
        continue
    fi
    
    echo "[INFO] Downloading ${model_name}..."
    $PYTHON -c "
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = '${HF_CACHE_REPO}'
subfolder = '${model_name}'
local_dir = '${local_dir}'

Path(local_dir).mkdir(parents=True, exist_ok=True)

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type='model',
        allow_patterns=[f'{subfolder}/*']
    )
    
    # Move files from subfolder to local_dir root
    subfolder_path = Path(local_dir) / subfolder
    if subfolder_path.exists():
        for item in subfolder_path.iterdir():
            target = Path(local_dir) / item.name
            if target.exists():
                if target.is_dir():
                    import shutil
                    shutil.rmtree(target)
                else:
                    target.unlink()
            item.rename(target)
        subfolder_path.rmdir()
    print(f'[INFO] Downloaded {subfolder}')
except Exception as e:
    print(f'[WARN] Failed to download {subfolder}: {e}')
"
done

# Cleanup temp files
rm -rf "$TMP_DIR"

# ============================================
# PHASE 3: Package
# ============================================

echo ""
echo "=========================================="
echo "  PHASE 3: Creating Package"
echo "=========================================="

echo ""
echo "[8/8] Creating package..."

# Cleanup unnecessary files
rm -rf "${SRC_DIR}/.github"
rm -rf "${SRC_DIR}/Docker"
rm -rf "${SRC_DIR}/docs"
rm -f "${SRC_DIR}/.gitignore"
rm -f "${SRC_DIR}/.dockerignore"
rm -f "${SRC_DIR}/README.md"
rm -f "${SRC_DIR}"/*.bat
rm -f "${SRC_DIR}"/*.ps1
rm -f "${SRC_DIR}"/*.ipynb

# Make startup scripts executable
chmod +x "${SRC_DIR}/go-webui.sh"
chmod +x "${SRC_DIR}/go-api.sh"

# Create symlink for packaging
cd ..
ln -sfn "$(basename "$SRC_DIR")" "$PKG_NAME"

TAR_ZST_PATH="${PKG_NAME}.tar.zst"

echo "[INFO] Compressing to ${TAR_ZST_PATH}..."
START_TIME=$(date +%s)

# Compress with zstd (-h follows symlinks)
tar -chf - "$PKG_NAME" | zstd -3 -T0 -o "$TAR_ZST_PATH"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[INFO] Compression completed in $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"

# Cleanup symlink
rm -f "$PKG_NAME"

# Show file info
PKG_SIZE=$(du -h "$TAR_ZST_PATH" | cut -f1)
echo "[INFO] Created package: ${TAR_ZST_PATH} (${PKG_SIZE})"

echo ""
echo "=========================================="
echo "  SUCCESS: ${TAR_ZST_PATH} created!"
echo "=========================================="
