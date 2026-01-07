#!/bin/bash
#
# Micromamba 安裝腳本
#

set -e

MAMBA_ROOT="$HOME/miniconda3"

if [ -d "$MAMBA_ROOT" ]; then
    echo "[INFO] Micromamba already installed, skipping"
    exit 0
fi

echo "[INFO] Downloading Micromamba..."
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /tmp bin/micromamba

echo "[INFO] Installing Micromamba..."
mkdir -p "$MAMBA_ROOT/bin"
mv /tmp/bin/micromamba "$MAMBA_ROOT/bin/"
rm -rf /tmp/bin

export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
MAMBA="$MAMBA_ROOT/bin/micromamba"

# 初始化 shell
$MAMBA shell init -s bash --root-prefix "$MAMBA_ROOT" > /dev/null

# 安裝 Python
echo "[INFO] Installing Python 3.11..."
$MAMBA create -n base -y -q
$MAMBA install -n base python=3.11 -c conda-forge -y -q

# 建立相容性符號連結
ln -sf "$MAMBA_ROOT/envs/base/bin/python" "$MAMBA_ROOT/bin/python"
ln -sf "$MAMBA_ROOT/envs/base/bin/pip" "$MAMBA_ROOT/bin/pip"

# 建立 conda wrapper script (install 指令自動加上 -n base)
cat > "$MAMBA_ROOT/bin/conda" << 'WRAPPER'
#!/bin/bash
MAMBA_ROOT="$HOME/miniconda3"
MAMBA="$MAMBA_ROOT/bin/micromamba"

# 如果第一個參數是 install，自動加上 -n base
if [ "$1" = "install" ]; then
    shift
    exec "$MAMBA" install -n base "$@"
else
    exec "$MAMBA" "$@"
fi
WRAPPER
chmod +x "$MAMBA_ROOT/bin/conda"

# 建立 conda.sh 相容檔案
mkdir -p "$MAMBA_ROOT/etc/profile.d"
cat > "$MAMBA_ROOT/etc/profile.d/conda.sh" << 'EOF'
export MAMBA_ROOT_PREFIX="$HOME/miniconda3"
export PATH="$MAMBA_ROOT_PREFIX/envs/base/bin:$MAMBA_ROOT_PREFIX/bin:$PATH"
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)"
micromamba activate base
EOF

echo "[INFO] Micromamba setup completed"
