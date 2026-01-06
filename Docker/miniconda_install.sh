#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd "$SCRIPT_DIR" || exit 1

cd .. || exit 1

if [ -d "$HOME/miniconda3" ]; then
    exit 0
fi

WGET_CMD=(wget --tries=25 --wait=5 --read-timeout=40 --retry-on-http-error=404)

# Download latest Miniconda for Linux x86_64
"${WGET_CMD[@]}" -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

LOG_PATH="/tmp/miniconda-install.log"

bash miniconda.sh -b -p "$HOME/miniconda3" >"$LOG_PATH" 2>&1

if [ $? -eq 0 ]; then
    echo "== Miniconda Installed =="
else
    echo "Failed to Install miniconda"
    tail -n 50 "$LOG_PATH"
    exit 1
fi

rm miniconda.sh

source "$HOME/miniconda3/etc/profile.d/conda.sh"

"$HOME/miniconda3/bin/conda" config --add channels conda-forge

"$HOME/miniconda3/bin/conda" update -q --all -y 1>/dev/null

"$HOME/miniconda3/bin/conda" install python=3.11 -q -y

"$HOME/miniconda3/bin/conda" install gcc=14 gxx ffmpeg cmake make unzip -q -y

if [ "$CUDA_VERSION" = "12.8" ]; then
    "$HOME/miniconda3/bin/pip" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
elif [ "$CUDA_VERSION" = "12.6" ]; then
    "$HOME/miniconda3/bin/pip" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
fi

"$HOME/miniconda3/bin/pip" cache purge

rm $LOG_PATH

rm -rf "$HOME/miniconda3/pkgs"

mkdir -p "$HOME/miniconda3/pkgs"

rm -rf "$HOME/.conda" "$HOME/.cache"
