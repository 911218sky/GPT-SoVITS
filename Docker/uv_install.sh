#!/bin/bash
#
# uv 安裝腳本 - 建立 Python 環境
#

set -e

UV_ROOT="$HOME/uv"
ENV_PATH="$UV_ROOT/env"

if [ -d "$ENV_PATH" ]; then
    echo "[INFO] uv environment already exists, skipping"
    exit 0
fi

echo "[INFO] Downloading and installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$UV_ROOT" sh

UV="$UV_ROOT/uv"

# 建立 Python 環境
echo "[INFO] Creating Python 3.11 environment..."
$UV venv "$ENV_PATH" --python 3.11

# 建立相容性符號連結
mkdir -p "$UV_ROOT/bin"
ln -sf "$ENV_PATH/bin/python" "$UV_ROOT/bin/python"
ln -sf "$ENV_PATH/bin/pip" "$UV_ROOT/bin/pip"
ln -sf "$UV" "$UV_ROOT/bin/uv"

# 建立 profile 檔案
mkdir -p "$UV_ROOT/etc/profile.d"
cat > "$UV_ROOT/etc/profile.d/uv.sh" << EOF
export PATH="$ENV_PATH/bin:$UV_ROOT/bin:$UV_ROOT:\$PATH"
export UV_PYTHON="$ENV_PATH/bin/python"
export VIRTUAL_ENV="$ENV_PATH"
EOF

echo "[INFO] uv setup completed"
