$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "  GPT-SoVITS Windows Package Builder"
Write-Host "=========================================="

# === Configuration ===
$cuda = $env:TORCH_CUDA
if (-not $cuda) {
    Write-Error "Missing TORCH_CUDA env (cu124 or cu128)"
    exit 1
}

$date = $env:DATE_SUFFIX
if ([string]::IsNullOrWhiteSpace($date)) {
    $date = Get-Date -Format "MMdd"
}

$suffix = $env:PKG_SUFFIX
$pkgName = "GPT-SoVITS-$date"
if (-not [string]::IsNullOrWhiteSpace($suffix)) {
    $pkgName = "$pkgName$suffix"
}
$pkgName = "$pkgName-$cuda"

$srcDir = $PWD
$tmpDir = "$srcDir\tmp"

# Set short temp path to avoid Windows path length issues
$env:TMPDIR = "C:\tmp"
$env:TEMP = "C:\tmp"
$env:TMP = "C:\tmp"
New-Item -ItemType Directory -Force -Path "C:\tmp" | Out-Null

# HuggingFace base URL
$HF_BASE = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main"

Write-Host "[INFO] Package: $pkgName"
Write-Host "[INFO] CUDA: $cuda"

# === Helper Functions ===
function Invoke-WithRetry {
    param(
        [scriptblock]$ScriptBlock,
        [int]$MaxRetries = 15,
        [int]$DelaySeconds = 5,
        [string]$OperationName = "Operation"
    )
    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            Start-Sleep -Seconds $DelaySeconds
            & $ScriptBlock
            return $true
        }
        catch {
            if ($i -lt $MaxRetries) {
                Write-Host "[INFO] $OperationName attempt $i failed, retrying..."
            }
        }
    }
    Write-Host "[WARNING] $OperationName failed after $MaxRetries attempts"
    return $false
}

function Save-HFFolder($repoPath, $localDir) {
    $apiUrl = "https://huggingface.co/api/models/lj1995/GPT-SoVITS/tree/main/$repoPath"
    $files = Invoke-RestMethod -Uri $apiUrl
    New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    foreach ($file in $files) {
        if ($file.type -eq "file") {
            $fileName = Split-Path $file.path -Leaf
            $downloadUrl = "$HF_BASE/$($file.path)"
            $destPath = Join-Path $localDir $fileName
            Write-Host "  -> $fileName"
            Invoke-WebRequest -Uri $downloadUrl -OutFile $destPath
        }
    }
}

function Save-ModelScopeModel($modelId, $localDir) {
    $modelName = $modelId -replace ".*/", ""
    $targetDir = "$localDir\$modelName"
    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
    
    # Get file list from ModelScope API
    $apiUrl = "https://www.modelscope.cn/api/v1/models/$modelId/repo/files"
    try {
        $response = Invoke-RestMethod -Uri $apiUrl
        foreach ($file in $response.Data.Files) {
            if ($file.Type -eq "file" -and $file.Path -notmatch "^(fig|example)/") {
                $fileName = $file.Path
                $downloadUrl = "https://www.modelscope.cn/models/$modelId/resolve/master/$fileName"
                Write-Host "  -> $fileName"
                Invoke-WebRequest -Uri $downloadUrl -OutFile "$targetDir\$fileName"
            }
        }
    }
    catch {
        # Fallback: download known required files
        Write-Host "  [WARN] API failed, using fallback file list"
        $baseUrl = "https://www.modelscope.cn/models/$modelId/resolve/master"
        $defaultFiles = @("configuration.json", "config.yaml", "model.pt", "README.md")
        foreach ($file in $defaultFiles) {
            try {
                Write-Host "  -> $file"
                Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile "$targetDir\$file"
            }
            catch {}
        }
    }
}

# === Cleanup ===
Write-Host "`n[1/9] Cleaning up..."
Remove-Item "$srcDir\.git" -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

# ============================================
# PHASE 1: Download all resources (no Python needed)
# ============================================

Write-Host "`n=========================================="
Write-Host "  PHASE 1: Downloading Resources"
Write-Host "=========================================="

# === Download FFmpeg ===
Write-Host "`n[2/9] Downloading FFmpeg..."
$ffUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$ffZip = "$tmpDir\ffmpeg.zip"
Invoke-WebRequest -Uri $ffUrl -OutFile $ffZip

# === Download Pretrained Models ===
Write-Host "`n[3/9] Downloading pretrained models from HuggingFace..."
$pretrainedDir = "$srcDir\GPT_SoVITS\pretrained_models"
New-Item -ItemType Directory -Force -Path $pretrainedDir | Out-Null

Write-Host "[INFO] Downloading gsv-v2final-pretrained..."
Save-HFFolder "gsv-v2final-pretrained" "$pretrainedDir\gsv-v2final-pretrained"

Write-Host "[INFO] Downloading chinese-hubert-base..."
Save-HFFolder "chinese-hubert-base" "$pretrainedDir\chinese-hubert-base"

Write-Host "[INFO] Downloading chinese-roberta-wwm-ext-large..."
Save-HFFolder "chinese-roberta-wwm-ext-large" "$pretrainedDir\chinese-roberta-wwm-ext-large"

Write-Host "[INFO] Downloading v2Pro..."
Save-HFFolder "v2Pro" "$pretrainedDir\v2Pro"

Write-Host "[INFO] Downloading sv model..."
New-Item -ItemType Directory -Force -Path "$pretrainedDir\sv" | Out-Null
Invoke-WebRequest -Uri "$HF_BASE/sv/pretrained_eres2netv2w24s4ep4.ckpt" -OutFile "$pretrainedDir\sv\pretrained_eres2netv2w24s4ep4.ckpt"

Write-Host "[INFO] Downloading s1v3.ckpt..."
Invoke-WebRequest -Uri "$HF_BASE/s1v3.ckpt" -OutFile "$pretrainedDir\s1v3.ckpt"

# === Download G2PW Model ===
Write-Host "`n[4/9] Downloading G2PW model..."
$g2pwUrl = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
$g2pwZip = "$tmpDir\G2PWModel.zip"
Invoke-WebRequest -Uri $g2pwUrl -OutFile $g2pwZip
Expand-Archive -Path $g2pwZip -DestinationPath "$srcDir\GPT_SoVITS\text" -Force
Remove-Item $g2pwZip

Write-Host "`n[INFO] All resources downloaded!"

# ============================================
# PHASE 2: Setup Python Environment
# ============================================

Write-Host "`n=========================================="
Write-Host "  PHASE 2: Setting up Environment"
Write-Host "=========================================="

# === Install Miniconda ===
Write-Host "`n[6/9] Installing Miniconda..."
$condaInstaller = "$tmpDir\miniconda.exe"
$condaPath = "$srcDir\runtime"

Invoke-WebRequest "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $condaInstaller
Start-Process -FilePath $condaInstaller -ArgumentList "/S", "/D=$condaPath" -Wait
Remove-Item $condaInstaller

$conda = "$condaPath\Scripts\conda.exe"
$pip = "$condaPath\Scripts\pip.exe"
$python = "$condaPath\python.exe"

Write-Host "[INFO] Setting up Python 3.11..."
& $conda install python=3.11 -y -q
& $conda clean -afy | Out-Null

# === Install PyTorch ===
Write-Host "`n[7/9] Installing PyTorch ($cuda)..."
switch ($cuda) {
    "cu126" {
        & $pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    }
    "cu128" {
        & $pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    }
    default {
        Write-Error "Unsupported CUDA version: $cuda"
        exit 1
    }
}

# === Install Dependencies ===
Write-Host "`n[8/9] Installing dependencies..."
& $pip install -r requirements.txt --no-warn-script-location
& $pip install -r extra-req.txt --no-warn-script-location
& $pip cache purge

# Extract FFmpeg
Write-Host "[INFO] Extracting FFmpeg..."
Expand-Archive $ffZip -DestinationPath $tmpDir -Force
$ffDir = Get-ChildItem -Directory "$tmpDir" | Where-Object { $_.Name -like "ffmpeg*" } | Select-Object -First 1
Copy-Item "$($ffDir.FullName)\bin\ffmpeg.exe" "$condaPath" -Force
Copy-Item "$($ffDir.FullName)\bin\ffprobe.exe" "$condaPath" -Force
Remove-Item $ffZip
Remove-Item $ffDir.FullName -Recurse -Force

# Download NLTK Data
Write-Host "[INFO] Downloading NLTK data..."
& $python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# Download FunASR models using ModelScope SDK (complete downloads)
Write-Host "[INFO] Downloading FunASR models via ModelScope SDK..."
& $pip install modelscope -q --no-warn-script-location

$asrModelsDir = "$srcDir\tools\asr\models"
$modelscriptContent = @"
from modelscope import snapshot_download
import os
import time

models_dir = r'$asrModelsDir'
os.makedirs(models_dir, exist_ok=True)

models = [
    'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', 
    'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
]

def download_with_retry(model_id, local_dir, max_retries=5):
    for i in range(max_retries):
        try:
            snapshot_download(model_id, local_dir=local_dir)
            return
        except Exception as e:
            print(f"[WARN] Attempt {i+1} failed for {model_id}: {e}")
            if i < max_retries - 1:
                print("Retrying in 10 seconds...")
                time.sleep(10)
            else:
                raise e

for model_id in models:
    model_name = model_id.split('/')[-1]
    local_dir = os.path.join(models_dir, model_name)
    print(f'[INFO] Downloading {model_name}...')
    download_with_retry(model_id, local_dir)
    print(f'[INFO] Downloaded {model_name}')

print('[INFO] All FunASR models downloaded successfully!')
"@

$modelscriptContent | Out-File -FilePath "$tmpDir\download_funasr.py" -Encoding UTF8
& $python "$tmpDir\download_funasr.py"
Remove-Item "$tmpDir\download_funasr.py" -Force

# ============================================
# PHASE 3: Package
# ============================================

Write-Host "`n=========================================="
Write-Host "  PHASE 3: Creating Package"
Write-Host "=========================================="

Write-Host "`n[9/9] Creating package..."

# Cleanup unnecessary files
$removeItems = @(
    "$tmpDir",
    "$srcDir\.github",
    "$srcDir\Docker", 
    "$srcDir\docs",
    "$srcDir\.gitignore",
    "$srcDir\.dockerignore",
    "$srcDir\README.md"
)
foreach ($item in $removeItems) {
    Remove-Item $item -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem "$srcDir" -Filter "*.sh" | Remove-Item -Force
Get-ChildItem "$srcDir" -Filter "*.ipynb" | Remove-Item -Force

# Create Junction for correct folder name in 7z
Set-Location ..
$junctionName = $pkgName
$junctionTarget = $srcDir

Write-Host "[INFO] Creating Junction: $junctionName -> $junctionTarget"
if (Test-Path $junctionName) {
    if ((Get-Item $junctionName).Attributes -match "ReparsePoint") {
        cmd /c rmdir "$junctionName"
    }
    else {
        Remove-Item "$junctionName" -Recurse -Force
    }
}
cmd /c mklink /J "$junctionName" "$junctionTarget"

$7zPath = "$pkgName.7z"

Write-Host "[INFO] Compressing to $7zPath..."
$start = Get-Date

# Compress the JUNCTION, not the source dir directly, to get the correct folder name in archive
& "C:\Program Files\7-Zip\7z.exe" a -t7z "$7zPath" "$junctionName" -m0=lzma2 -mx=3 -mmt=on -bsp1

$end = Get-Date
Write-Host "[INFO] Compression completed in $([math]::Round(($end - $start).TotalMinutes, 2)) minutes"

# Cleanup Junction
Write-Host "[INFO] Removing Junction..."
cmd /c rmdir "$junctionName"

# Show file info
$pkgFile = Get-Item "$7zPath"
Write-Host "[INFO] Created package: $($pkgFile.Name) ($([math]::Round($pkgFile.Length / 1GB, 2)) GB)"

Write-Host ""
Write-Host "=========================================="
Write-Host "  SUCCESS: $7zPath created!"
Write-Host "=========================================="
