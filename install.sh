#!/bin/bash

# cd into GPT-SoVITS Base Path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

RESET="\033[0m"
BOLD="\033[1m"
ERROR="\033[1;31m[ERROR]: $RESET"
WARNING="\033[1;33m[WARNING]: $RESET"
INFO="\033[1;32m[INFO]: $RESET"
SUCCESS="\033[1;34m[SUCCESS]: $RESET"

set -eE
set -o errtrace

trap 'on_error $LINENO "$BASH_COMMAND" $?' ERR

# shellcheck disable=SC2317
on_error() {
    local lineno="$1"
    local cmd="$2"
    local code="$3"

    echo -e "${ERROR}${BOLD}Command \"${cmd}\" Failed${RESET} at ${BOLD}Line ${lineno}${RESET} with Exit Code ${BOLD}${code}${RESET}"
    echo -e "${ERROR}${BOLD}Call Stack:${RESET}"
    for ((i = ${#FUNCNAME[@]} - 1; i >= 1; i--)); do
        echo -e "  in ${BOLD}${FUNCNAME[i]}()${RESET} at ${BASH_SOURCE[i]}:${BOLD}${BASH_LINENO[i - 1]}${RESET}"
    done
    exit "$code"
}

run_uv_quiet() {
    local output
    output=$(uv pip install "$@" 2>&1) || {
        echo -e "${ERROR} uv pip install failed:\n$output"
        exit 1
    }
}

run_wget_quiet() {
    # Use --show-progress only if TTY is available, otherwise use quiet mode
    local wget_opts="--tries=25 --wait=5 --read-timeout=40"
    if [ -t 1 ]; then
        wget_opts="$wget_opts -q --show-progress"
    else
        wget_opts="$wget_opts -nv"
    fi
    if wget $wget_opts "$@" 2>&1; then
        # tput may fail in non-TTY environments (like Docker), ignore errors
        (tput cuu1 && tput el) 2>/dev/null || true
    else
        echo -e "${ERROR} Wget failed: $*"
        exit 1
    fi
}

# Check for uv
if ! command -v uv &>/dev/null; then
    echo -e "${ERROR}uv Not Found. Please install uv first:"
    echo -e "${INFO}  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

USE_CUDA=false
USE_ROCM=false
USE_CPU=false

USE_HF=false
USE_HF_MIRROR=false
USE_MODELSCOPE=false
DOWNLOAD_UVR5=false

print_help() {
    echo "Usage: bash install.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --device   CU118|CU126|CU128|ROCM|MPS|CPU    Specify the Device (REQUIRED)"
    echo "  --source   HF|HF-Mirror|ModelScope     Specify the model source (REQUIRED)"
    echo "  --download-uvr5                        Enable downloading the UVR5 model"
    echo "  -h, --help                             Show this help message and exit"
    echo ""
    echo "Examples:"
    echo "  bash install.sh --device CU128 --source HF --download-uvr5"
    echo "  bash install.sh --device MPS --source ModelScope"
}

# Show help if no arguments provided
if [[ $# -eq 0 ]]; then
    print_help
    exit 0
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
    --source)
        case "$2" in
        HF)
            USE_HF=true
            ;;
        HF-Mirror)
            USE_HF_MIRROR=true
            ;;
        ModelScope)
            USE_MODELSCOPE=true
            ;;
        *)
            echo -e "${ERROR}Error: Invalid Download Source: $2"
            echo -e "${ERROR}Choose From: [HF, HF-Mirror, ModelScope]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --device)
        case "$2" in
        CU118)
            CUDA=118
            USE_CUDA=true
            ;;
        CU126)
            CUDA=126
            USE_CUDA=true
            ;;
        CU128)
            CUDA=128
            USE_CUDA=true
            ;;
        ROCM)
            USE_ROCM=true
            ;;
        MPS)
            USE_CPU=true
            ;;
        CPU)
            USE_CPU=true
            ;;
        *)
            echo -e "${ERROR}Error: Invalid Device: $2"
            echo -e "${ERROR}Choose From: [CU118, CU126, CU128, ROCM, MPS, CPU]"
            exit 1
            ;;
        esac
        shift 2
        ;;
    --download-uvr5)
        DOWNLOAD_UVR5=true
        shift
        ;;
    -h | --help)
        print_help
        exit 0
        ;;
    *)
        echo -e "${ERROR}Unknown Argument: $1"
        echo ""
        print_help
        exit 1
        ;;
    esac
done

if ! $USE_CUDA && ! $USE_ROCM && ! $USE_CPU; then
    echo -e "${ERROR}Error: Device is REQUIRED"
    echo ""
    print_help
    exit 1
fi

if ! $USE_HF && ! $USE_HF_MIRROR && ! $USE_MODELSCOPE; then
    echo -e "${ERROR}Error: Download Source is REQUIRED"
    echo ""
    print_help
    exit 1
fi

# Check system requirements
echo -e "${INFO}Detected system: $(uname -s) $(uname -r) $(uname -m)"

# Check for required system tools
if [ "$(uname)" != "Darwin" ]; then
    # Linux: check for gcc
    gcc_major_version=$(command -v gcc >/dev/null 2>&1 && gcc -dumpversion | cut -d. -f1 || echo 0)
    if [ "$gcc_major_version" -lt 11 ]; then
        echo -e "${WARNING}GCC version < 11 detected. Some packages may fail to compile."
        echo -e "${WARNING}Please install GCC 11+ using your system package manager:"
        echo -e "${INFO}  Ubuntu/Debian: sudo apt install gcc-11 g++-11"
        echo -e "${INFO}  Fedora: sudo dnf install gcc gcc-c++"
    else
        echo -e "${INFO}Detected GCC Version: $gcc_major_version"
    fi
else
    # macOS: check for Xcode
    if ! xcode-select -p &>/dev/null; then
        echo -e "${INFO}Installing Xcode Command Line Tools..."
        xcode-select --install
        echo -e "${INFO}Waiting For Xcode Command Line Tools Installation Complete..."
        while true; do
            sleep 20
            if xcode-select -p &>/dev/null; then
                echo -e "${SUCCESS}Xcode Command Line Tools Installed"
                break
            else
                echo -e "${INFO}Installing，Please Wait..."
            fi
        done
    else
        XCODE_PATH=$(xcode-select -p)
        if [[ "$XCODE_PATH" == *"Xcode.app"* ]]; then
            echo -e "${WARNING} Detected Xcode path: $XCODE_PATH"
            echo -e "${WARNING} If your Xcode version does not match your macOS version, it may cause unexpected issues during compilation or package builds."
        fi
    fi
fi

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo -e "${WARNING}FFmpeg not found. Please install it:"
    echo -e "${INFO}  Ubuntu/Debian: sudo apt install ffmpeg"
    echo -e "${INFO}  macOS: brew install ffmpeg"
    echo -e "${INFO}  Fedora: sudo dnf install ffmpeg"
fi

# Check for unzip
if ! command -v unzip &>/dev/null; then
    echo -e "${ERROR}unzip not found. Please install it:"
    echo -e "${INFO}  Ubuntu/Debian: sudo apt install unzip"
    echo -e "${INFO}  macOS: brew install unzip"
    exit 1
fi

if [ "$USE_HF" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace"
    PRETRINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
    NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_HF_MIRROR" = "true" ]; then
    echo -e "${INFO}Download Model From HuggingFace-Mirror"
    PRETRINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
    G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
    UVR5_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
    NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
    PYOPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
elif [ "$USE_MODELSCOPE" = "true" ]; then
    echo -e "${INFO}Download Model From ModelScope"
    PRETRINED_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
    G2PW_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
    UVR5_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip"
    NLTK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
    PYOPENJTALK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
fi

if [ ! -d "GPT_SoVITS/pretrained_models/sv" ]; then
    echo -e "${INFO}Downloading Pretrained Models..."
    rm -rf pretrained_models.zip
    run_wget_quiet "$PRETRINED_URL"

    mkdir -p GPT_SoVITS
    unzip -q -o pretrained_models.zip -d GPT_SoVITS
    rm -rf pretrained_models.zip
    echo -e "${SUCCESS}Pretrained Models Downloaded"
else
    echo -e "${INFO}Pretrained Model Exists"
    echo -e "${INFO}Skip Downloading Pretrained Models"
fi

if [ ! -d "GPT_SoVITS/text/G2PWModel" ]; then
    echo -e "${INFO}Downloading G2PWModel.."
    rm -rf G2PWModel.zip
    run_wget_quiet "$G2PW_URL"

    mkdir -p GPT_SoVITS/text
    unzip -q -o G2PWModel.zip -d GPT_SoVITS/text
    rm -rf G2PWModel.zip
    echo -e "${SUCCESS}G2PWModel Downloaded"
else
    echo -e "${INFO}G2PWModel Exists"
    echo -e "${INFO}Skip Downloading G2PWModel"
fi

if [ "$DOWNLOAD_UVR5" = "true" ]; then
    if [ -d "tools/uvr5/uvr5_weights" ] && find -L "tools/uvr5/uvr5_weights" -mindepth 1 ! -name '.gitignore' 2>/dev/null | grep -q .; then
        echo -e "${INFO}UVR5 Models Exists"
        echo -e "${INFO}Skip Downloading UVR5 Models"
    else
        echo -e "${INFO}Downloading UVR5 Models..."
        rm -rf uvr5_weights.zip
        run_wget_quiet "$UVR5_URL"

        mkdir -p tools/uvr5
        unzip -q -o uvr5_weights.zip -d tools/uvr5
        rm -rf uvr5_weights.zip
        echo -e "${SUCCESS}UVR5 Models Downloaded"
    fi
fi

# Install PyTorch based on device selection
# PyTorch version configuration
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"

if [ "$USE_CUDA" = true ]; then
    if [ "$CUDA" = 128 ]; then
        echo -e "${INFO}Installing PyTorch ${TORCH_VERSION} For CUDA 12.8..."
        run_uv_quiet torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url "https://download.pytorch.org/whl/cu128"
    elif [ "$CUDA" = 126 ]; then
        echo -e "${INFO}Installing PyTorch ${TORCH_VERSION} For CUDA 12.6..."
        run_uv_quiet torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url "https://download.pytorch.org/whl/cu126"
    elif [ "$CUDA" = 118 ]; then
        echo -e "${INFO}Installing PyTorch ${TORCH_VERSION} For CUDA 11.8..."
        run_uv_quiet torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url "https://download.pytorch.org/whl/cu118"
    fi
elif [ "$USE_ROCM" = true ]; then
    echo -e "${INFO}Installing PyTorch ${TORCH_VERSION} For ROCm 6.3..."
    run_uv_quiet torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url "https://download.pytorch.org/whl/rocm6.3"
elif [ "$USE_CPU" = true ]; then
    echo -e "${INFO}Installing PyTorch ${TORCH_VERSION} For CPU..."
    run_uv_quiet torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url "https://download.pytorch.org/whl/cpu"
fi
echo -e "${SUCCESS}PyTorch Installed"

echo -e "${INFO}Installing Python Dependencies From requirements.txt..."

# Install main requirements first (includes onnxruntime-gpu needed by faster-whisper)
run_uv_quiet -r requirements.txt

# Install extra requirements (faster-whisper) - it will use already installed onnxruntime-gpu
run_uv_quiet -r extra-req.txt

echo -e "${SUCCESS}Python Dependencies Installed"

PY_PREFIX=$(python -c "import sys; print(sys.prefix)")
PYOPENJTALK_PREFIX=$(python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))")

echo -e "${INFO}Downloading NLTK Data..."
rm -rf nltk_data.zip
run_wget_quiet "$NLTK_URL" -O nltk_data.zip
unzip -q -o nltk_data -d "$PY_PREFIX"
rm -rf nltk_data.zip
echo -e "${SUCCESS}NLTK Data Downloaded"

echo -e "${INFO}Downloading Open JTalk Dict..."
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
run_wget_quiet "$PYOPENJTALK_URL" -O open_jtalk_dic_utf_8-1.11.tar.gz
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C "$PYOPENJTALK_PREFIX"
rm -rf open_jtalk_dic_utf_8-1.11.tar.gz
echo -e "${SUCCESS}Open JTalk Dic Downloaded"

if [ "$USE_ROCM" = true ] && [ "$IS_WSL" = true ]; then
    echo -e "${INFO}Updating WSL Compatible Runtime Lib For ROCm..."
    location=$(uv pip show torch | grep Location | awk -F ": " '{print $2}')
    cd "${location}"/torch/lib/ || exit
    rm libhsa-runtime64.so*
    cp "$(readlink -f /opt/rocm/lib/libhsa-runtime64.so)" libhsa-runtime64.so
    echo -e "${SUCCESS}ROCm Runtime Lib Updated..."
fi

echo -e "${SUCCESS}Installation Completed"
