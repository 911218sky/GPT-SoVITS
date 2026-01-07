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

# === Install Micromamba ===
Write-Host "`n[6/9] Installing Micromamba..."
$runtimePath = "$srcDir\runtime"
$mambaExe = "$runtimePath\micromamba.exe"

New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
Invoke-WebRequest "https://micro.mamba.pm/api/micromamba/win-64/latest" -OutFile "$tmpDir\micromamba.tar.bz2"

# Extract micromamba
tar -xf "$tmpDir\micromamba.tar.bz2" -C $tmpDir
Copy-Item "$tmpDir\Library\bin\micromamba.exe" $mambaExe -Force
Remove-Item "$tmpDir\micromamba.tar.bz2" -Force
Remove-Item "$tmpDir\Library" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$tmpDir\info" -Recurse -Force -ErrorAction SilentlyContinue

# Setup micromamba environment
# Use --prefix to create environment at specific path (avoids root prefix issues)
$envPath = "$runtimePath\env"
Write-Host "[DEBUG] Creating environment at: $envPath"
& $mambaExe create --prefix $envPath python=3.11 -c conda-forge -y -q
& $mambaExe clean -afy 2>$null | Out-Null

$pip = "$envPath\Scripts\pip.exe"
$python = "$envPath\python.exe"

# Verify pip exists
if (-not (Test-Path $pip)) {
    Write-Error "pip not found at $pip"
    Write-Host "[DEBUG] Listing $runtimePath contents:"
    Get-ChildItem $runtimePath -Recurse -Depth 2 | Select-Object FullName
    exit 1
}

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
# Remove --no-binary constraint for faster installation
# Create temporary requirements without --no-binary
$reqContent = Get-Content "requirements.txt" | Where-Object { $_ -notmatch "^--no-binary" }
$reqContent | Out-File "requirements_optimized.txt" -Encoding UTF8

# Install with optimizations
& $pip install -r requirements_optimized.txt --no-warn-script-location -v
& $pip install -r extra-req.txt --no-warn-script-location -v

Remove-Item "requirements_optimized.txt" -Force

# Cleanup caches to reduce package size
Write-Host "[INFO] Cleaning up caches..."
& $pip cache purge
& $mambaExe clean -afy | Out-Null
Remove-Item "$runtimePath\pkgs" -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path "$runtimePath\pkgs" | Out-Null
Remove-Item "$env:USERPROFILE\.cache" -Recurse -Force -ErrorAction SilentlyContinue

# Remove unnecessary files from site-packages
$sitePackages = "$envPath\Lib\site-packages"
Get-ChildItem $sitePackages -Recurse -Include "*.pyc", "*.pyo" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem $sitePackages -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $sitePackages -Recurse -Directory -Filter "tests" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $sitePackages -Recurse -Directory -Filter "test" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Clean up dist-info directories (keep only essential files)
$distInfoDirs = Get-ChildItem $sitePackages -Directory -Filter "*.dist-info" -ErrorAction SilentlyContinue
if ($distInfoDirs) {
    foreach ($dir in $distInfoDirs) {
        if ($dir -and $dir.FullName -and (Test-Path $dir.FullName)) {
            Get-ChildItem $dir.FullName -Exclude "METADATA", "RECORD", "WHEEL", "entry_points.txt", "top_level.txt" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
        }
    }
}

# Remove Triton (not used on Windows)
Remove-Item "$sitePackages\triton*" -Recurse -Force -ErrorAction SilentlyContinue

# Extract FFmpeg
Write-Host "[INFO] Extracting FFmpeg..."
Expand-Archive $ffZip -DestinationPath $tmpDir -Force
$ffDir = Get-ChildItem -Directory "$tmpDir" | Where-Object { $_.Name -like "ffmpeg*" } | Select-Object -First 1
Copy-Item "$($ffDir.FullName)\bin\ffmpeg.exe" "$envPath" -Force
Copy-Item "$($ffDir.FullName)\bin\ffprobe.exe" "$envPath" -Force
Remove-Item $ffDir.FullName -Recurse -Force

# Download NLTK Data
Write-Host "[INFO] Downloading NLTK data..."
& $python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# Download fast-langdetect model (lid.176.bin)
Write-Host "[INFO] Downloading fast-langdetect model..."
$fastLangDetectDir = "$srcDir\GPT_SoVITS\pretrained_models\fast_langdetect"
New-Item -ItemType Directory -Force -Path $fastLangDetectDir | Out-Null
$lidModelUrl = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
$lidModelPath = "$fastLangDetectDir\lid.176.bin"
Write-Host "[INFO] Downloading lid.176.bin (125MB)..."
Invoke-WebRequest -Uri $lidModelUrl -OutFile $lidModelPath
Write-Host "[INFO] Downloaded fast-langdetect model"

# Download FunASR models (try HuggingFace cache first, fallback to ModelScope)
Write-Host "[INFO] Downloading FunASR models..."
& $pip install huggingface_hub modelscope -q --no-warn-script-location

$asrModelsDir = "$srcDir\tools\asr\models"
New-Item -ItemType Directory -Force -Path $asrModelsDir | Out-Null

$HF_CACHE_REPO = "sky1218/GPT-SoVITS-Models"
$funasr_models = @(
    @{ name = "speech_fsmn_vad_zh-cn-16k-common-pytorch"; modelscope_id = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch" },
    @{ name = "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"; modelscope_id = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch" },
    @{ name = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"; modelscope_id = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch" }
)

$huggingface_cli = "$envPath\Scripts\huggingface-cli.exe"
$modelscope_cli = "$envPath\Scripts\modelscope.exe"

foreach ($model in $funasr_models) {
    $modelName = $model.name
    $modelscopeId = $model.modelscope_id
    $localDir = "$asrModelsDir\$modelName"
    
    if ((Test-Path $localDir) -and (Get-ChildItem $localDir -ErrorAction SilentlyContinue)) {
        Write-Host "[INFO] $modelName already exists, skipping"
        continue
    }
    
    $downloaded = $false
    
    # Try HuggingFace first
    Write-Host "[INFO] Trying HuggingFace cache for $modelName..."
    $tmpDir_hf = "${localDir}_tmp"
    try {
        & $huggingface_cli download $HF_CACHE_REPO --include "$modelName/*" --local-dir $tmpDir_hf --quiet 2>$null
        $srcPath = "$tmpDir_hf\$modelName"
        if ((Test-Path $srcPath) -and (Get-ChildItem $srcPath -ErrorAction SilentlyContinue)) {
            if (Test-Path $localDir) { Remove-Item $localDir -Recurse -Force }
            Move-Item $srcPath $localDir -Force
            Remove-Item $tmpDir_hf -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "[INFO] Downloaded $modelName from HuggingFace cache"
            $downloaded = $true
        }
    } catch {
        Write-Host "[INFO] HuggingFace cache not available"
    }
    
    # Fallback to ModelScope
    if (-not $downloaded) {
        for ($i = 1; $i -le 5; $i++) {
            try {
                Write-Host "[INFO] Downloading $modelName from ModelScope (attempt $i)..."
                & $modelscope download --model $modelscopeId --local_dir $localDir 2>$null
                if ((Test-Path $localDir) -and (Get-ChildItem $localDir -ErrorAction SilentlyContinue)) {
                    Write-Host "[INFO] Downloaded $modelName from ModelScope"
                    $downloaded = $true
                    break
                }
            } catch {
                Write-Host "[WARN] Attempt $i failed: $_"
                if ($i -lt 5) {
                    Write-Host "Retrying in 10 seconds..."
                    Start-Sleep -Seconds 10
                }
            }
        }
    }
    
    if (-not $downloaded) {
        Write-Error "[ERROR] Failed to download $modelName"
        exit 1
    }
}

Write-Host "[INFO] All FunASR models downloaded successfully!"

# Clean up ModelScope cache files
Get-ChildItem "$asrModelsDir" -Recurse -Filter ".msc" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem "$asrModelsDir" -Recurse -Filter "*.lock" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem "$asrModelsDir" -Recurse -Filter "*.git*" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Clean up FFmpeg zip (downloaded earlier)
Remove-Item $ffZip -Force -ErrorAction SilentlyContinue

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

$tarZstPath = "$pkgName.tar.zst"

Write-Host "[INFO] Compressing to $tarZstPath..."
$start = Get-Date

# Download and setup zstd
$zstdDir = "zstd_tool"
$zstdPath = "$zstdDir\zstd.exe"
New-Item -ItemType Directory -Force -Path $zstdDir | Out-Null

if (-not (Test-Path $zstdPath)) {
    Write-Host "[INFO] Downloading zstd..."
    $zstdUrl = "https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-v1.5.6-win64.zip"
    $zstdZip = "$zstdDir\zstd.zip"
    Invoke-WebRequest -Uri $zstdUrl -OutFile $zstdZip
    Expand-Archive -Path $zstdZip -DestinationPath $zstdDir -Force
    Copy-Item "$zstdDir\zstd-v1.5.6-win64\zstd.exe" $zstdPath -Force
    Remove-Item $zstdZip -Force
    Remove-Item "$zstdDir\zstd-v1.5.6-win64" -Recurse -Force
    Write-Host "[INFO] zstd downloaded and ready"
}

# Create tar archive and compress with zstd
# -3: compression level 3 (good balance of speed and ratio)
# -T0: use all CPU cores for parallel compression
Write-Host "[INFO] Creating tar.zst archive..."
& tar -cf - "$junctionName" | & $zstdPath -3 -T0 -o "$tarZstPath"

$end = Get-Date
Write-Host "[INFO] Compression completed in $([math]::Round(($end - $start).TotalMinutes, 2)) minutes"

# Cleanup Junction
Write-Host "[INFO] Removing Junction..."
cmd /c rmdir "$junctionName"

# Cleanup zstd tool
Remove-Item "zstd_tool" -Recurse -Force -ErrorAction SilentlyContinue

# Show file info
$pkgFile = Get-Item "$tarZstPath"
Write-Host "[INFO] Created package: $($pkgFile.Name) ($([math]::Round($pkgFile.Length / 1GB, 2)) GB)"

Write-Host ""
Write-Host "=========================================="
Write-Host "  SUCCESS: $tarZstPath created!"
Write-Host "=========================================="