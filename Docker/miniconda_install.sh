#!/bin/bash
#
# Miniconda 安裝腳本
# 只負責安裝 Miniconda 和 Python
# 其他工具由 install.sh 統一管理
#

set -e

# 如果已安裝則跳過
if [ -d "$HOME/miniconda3" ]; then
    echo "[INFO] Miniconda already installed, skipping"
    exit 0
fi

echo "[INFO] Downloading Miniconda..."
wget -q --tries=25 --wait=5 --read-timeout=40 \
    -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

echo "[INFO] Installing Miniconda..."
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3" > /tmp/miniconda-install.log 2>&1
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Miniconda"
    tail -n 50 /tmp/miniconda-install.log
    exit 1
fi

# 清理安裝檔
rm -f /tmp/miniconda.sh /tmp/miniconda-install.log

# 安裝 Python（其他工具由 install.sh 處理）
echo "[INFO] Installing Python 3.11..."
"$HOME/miniconda3/bin/conda" install python=3.11 -q -y

echo "[INFO] Miniconda setup completed"
