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

# HuggingFace base URL (no auth required for public models)
$HF_BASE = "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main"

Write-Host "[INFO] Package: $pkgName"
Write-Host "[INFO] CUDA: $cuda"

# === Helper Functions ===
function Download-File($url, $dest) {
    $filename = Split-Path $url -Leaf
    Write-Host "  -> $filename"
    Invoke-WebRequest -Uri $url -OutFile $dest
}

function Download-HFFolder($repoPath, $localDir) {
    # Download all files from a HuggingFace folder
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

# === Cleanup ===
Write-Host "`n[1/8] Cleaning up..."
Remove-Item "$srcDir\.git" -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

# === Install Miniconda ===
Write-Host "`n[2/8] Installing Miniconda..."
$condaInstaller = "$tmpDir\miniconda.exe"
$condaPath = "$srcDir\runtime"

Invoke-WebRequest "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $condaInstaller
Start-Process -FilePath $condaInstaller -ArgumentList "/S", "/D=$condaPath" -Wait
Remove-Item $condaInstaller

$conda = "$condaPath\Scripts\conda.exe"
$pip = "$condaPath\Scripts\pip.exe"
$python = "$condaPath\python.exe"

# Initialize conda and install Python
Write-Host "[INFO] Setting up Python 3.11..."
& $conda install python=3.11 -y -q
& $conda clean -afy | Out-Null

# === Install PyTorch ===
Write-Host "`n[3/8] Installing PyTorch ($cuda)..."
switch ($cuda) {
    "cu124" {
        & $pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-warn-script-location
    }
    "cu128" {
        & $pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
    }
    default {
        Write-Error "Unsupported CUDA version: $cuda"
        exit 1
    }
}

# === Install Dependencies ===
Write-Host "`n[4/8] Installing dependencies..."
# Install main requirements first (includes onnxruntime-gpu needed by faster-whisper)
& $pip install -r requirements.txt --no-warn-script-location
# Install extra requirements (faster-whisper) - it will use already installed onnxruntime-gpu
& $pip install -r extra-req.txt --no-warn-script-location
& $pip cache purge

# === Download Models ===
Write-Host "`n[5/8] Downloading pretrained models from HuggingFace..."

$pretrainedDir = "$srcDir\GPT_SoVITS\pretrained_models"
New-Item -ItemType Directory -Force -Path $pretrainedDir | Out-Null

# Download gsv-v2final-pretrained models
Write-Host "[INFO] Downloading gsv-v2final-pretrained..."
Download-HFFolder "gsv-v2final-pretrained" "$pretrainedDir\gsv-v2final-pretrained"

# Download chinese-hubert-base
Write-Host "[INFO] Downloading chinese-hubert-base..."
Download-HFFolder "chinese-hubert-base" "$pretrainedDir\chinese-hubert-base"

# Download chinese-roberta-wwm-ext-large
Write-Host "[INFO] Downloading chinese-roberta-wwm-ext-large..."
Download-HFFolder "chinese-roberta-wwm-ext-large" "$pretrainedDir\chinese-roberta-wwm-ext-large"

# Download G2PW Model
Write-Host "[INFO] Downloading G2PW model..."
$g2pwUrl = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
$g2pwZip = "$tmpDir\G2PWModel.zip"
Invoke-WebRequest -Uri $g2pwUrl -OutFile $g2pwZip
Expand-Archive -Path $g2pwZip -DestinationPath "$srcDir\GPT_SoVITS\text" -Force
Remove-Item $g2pwZip

# === Download FFmpeg ===
Write-Host "`n[6/8] Downloading FFmpeg..."
$ffUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$ffZip = "$tmpDir\ffmpeg.zip"
Invoke-WebRequest -Uri $ffUrl -OutFile $ffZip
Expand-Archive $ffZip -DestinationPath $tmpDir -Force
$ffDir = Get-ChildItem -Directory "$tmpDir" | Where-Object { $_.Name -like "ffmpeg*" } | Select-Object -First 1
Copy-Item "$($ffDir.FullName)\bin\ffmpeg.exe" "$condaPath" -Force
Copy-Item "$($ffDir.FullName)\bin\ffprobe.exe" "$condaPath" -Force
Remove-Item $ffZip
Remove-Item $ffDir.FullName -Recurse -Force

# === Download NLTK Data ===
Write-Host "`n[7/8] Downloading NLTK data..."
& $python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# === Package ===
Write-Host "`n[8/8] Creating package..."

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

# Create package
Set-Location ..
$7zPath = "$pkgName.7z"

Write-Host "[INFO] Compressing to $7zPath (this may take a while)..."
$start = Get-Date
& "C:\Program Files\7-Zip\7z.exe" a -t7z "$7zPath" "$srcDir" -m0=lzma2 -mx=9 -mmt=on -bsp1
$end = Get-Date
Write-Host "[INFO] Compression completed in $([math]::Round(($end - $start).TotalMinutes, 2)) minutes"

# Rename folder with retry (Windows sometimes holds file handles briefly)
$maxRetries = 5
$retryDelay = 3
for ($i = 1; $i -le $maxRetries; $i++) {
    try {
        Start-Sleep -Seconds $retryDelay
        Rename-Item -Path $srcDir -NewName $pkgName -ErrorAction Stop
        Write-Host "[INFO] Folder renamed successfully"
        break
    }
    catch {
        if ($i -eq $maxRetries) {
            Write-Host "[WARNING] Could not rename folder after $maxRetries attempts, but package was created successfully"
        }
        else {
            Write-Host "[INFO] Rename attempt $i failed, retrying in $retryDelay seconds..."
        }
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  SUCCESS: $7zPath created!"
Write-Host "=========================================="
