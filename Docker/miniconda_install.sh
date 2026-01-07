#!/bin/bash
#
# Micromamba 安裝腳本 - 使用 --prefix 避免 root prefix 衝突
#

set -e

MAMBA_ROOT="$HOME/miniconda3"
ENV_PATH="$MAMBA_ROOT/env"

if [ -d "$ENV_PATH" ]; then
    echo "[INFO] Micromamba environment already exists, skipping"
    exit 0
fi

# 清除可能存在的設定檔
rm -rf "$HOME/.mambarc" "$HOME/.condarc" "$HOME/.mamba" 2>/dev/null || true

# 清除環境變數
unset MAMBA_ROOT_PREFIX
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

echo "[INFO] Downloading Micromamba..."
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba

echo "[INFO] Installing Micromamba..."
mkdir -p "$MAMBA_ROOT/bin"
mv /tmp/bin/micromamba "$MAMBA_ROOT/bin/"
rm -rf /tmp/bin

MAMBA="$MAMBA_ROOT/bin/micromamba"

# 使用 --prefix 直接指定環境路徑，避免 root prefix 問題
echo "[INFO] Installing Python 3.11..."
$MAMBA create --prefix "$ENV_PATH" python=3.11 -c conda-forge -y -q

# 建立相容性符號連結
ln -sf "$ENV_PATH/bin/python" "$MAMBA_ROOT/bin/python"
ln -sf "$ENV_PATH/bin/pip" "$MAMBA_ROOT/bin/pip"

# 建立 conda wrapper script (install 指令自動加上 --prefix)
cat > "$MAMBA_ROOT/bin/conda" << WRAPPER
#!/bin/bash
MAMBA_ROOT="$HOME/miniconda3"
ENV_PATH="\$MAMBA_ROOT/env"
MAMBA="\$MAMBA_ROOT/bin/micromamba"

# 如果第一個參數是 install，自動加上 --prefix
if [ "\$1" = "install" ]; then
    shift
    exec "\$MAMBA" install --prefix "\$ENV_PATH" "\$@"
else
    exec "\$MAMBA" "\$@"
fi
WRAPPER
chmod +x "$MAMBA_ROOT/bin/conda"

# 建立 conda.sh 相容檔案
mkdir -p "$MAMBA_ROOT/etc/profile.d"
cat > "$MAMBA_ROOT/etc/profile.d/conda.sh" << EOF
export PATH="$ENV_PATH/bin:$MAMBA_ROOT/bin:\$PATH"
EOF

echo "[INFO] Micromamba setup completed"
