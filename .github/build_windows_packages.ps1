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
Remove-Item $g2pwZip -Force -ErrorAction SilentlyContinue

Write-Host "`n[INFO] All resources downloaded!"

# ============================================
# PHASE 2: Setup Python Environment
# ============================================

Write-Host "`n=========================================="
Write-Host "  PHASE 2: Setting up Environment"
Write-Host "=========================================="

# === Install uv and Create Python Environment ===
Write-Host "`n[6/9] Installing uv and setting up portable Python environment..."
$runtimePath = "$srcDir\runtime"
$envPath = "$runtimePath\env"

New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
New-Item -ItemType Directory -Force -Path $envPath | Out-Null

# Download and install uv
Write-Host "[INFO] Downloading uv..."
$uvInstaller = "$tmpDir\uv-installer.ps1"
Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile $uvInstaller
$env:UV_INSTALL_DIR = $runtimePath
& powershell -ExecutionPolicy Bypass -File $uvInstaller
Remove-Item $uvInstaller -Force -ErrorAction SilentlyContinue

$uv = "$runtimePath\uv.exe"

# Verify uv exists
if (-not (Test-Path $uv)) {
    Write-Error "uv not found at $uv"
    exit 1
}

# Download standalone Python
Write-Host "[INFO] Downloading standalone Python 3.11..."
$pythonVersion = "3.11.12"
$pythonRelease = "20251217"
$pythonUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/$pythonRelease/cpython-$pythonVersion+$pythonRelease-x86_64-pc-windows-msvc-shared-install_only.tar.gz"
$pythonArchive = "$tmpDir\python.tar.gz"

Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonArchive

Write-Host "[INFO] Extracting Python..."
& tar -xzf $pythonArchive -C $tmpDir

# Move python folder contents to env
$pythonExtracted = "$tmpDir\python"
if (Test-Path $pythonExtracted) {
    Get-ChildItem $pythonExtracted | Move-Item -Destination $envPath -Force
}
Remove-Item $pythonArchive -Force -ErrorAction SilentlyContinue
Remove-Item $pythonExtracted -Recurse -Force -ErrorAction SilentlyContinue

$python = "$envPath\python.exe"

# Verify python exists
if (-not (Test-Path $python)) {
    Write-Error "python not found at $python"
    exit 1
}

# Remove EXTERNALLY-MANAGED marker to allow pip install
$externallyManaged = "$envPath\Lib\EXTERNALLY-MANAGED"
if (Test-Path $externallyManaged) {
    Remove-Item $externallyManaged -Force
}

Write-Host "[INFO] Standalone Python installed at: $python"

# === Install PyTorch ===
Write-Host "`n[7/9] Installing PyTorch ($cuda)..."
switch ($cuda) {
    "cu126" {
        & $uv pip install --python $python torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    }
    "cu128" {
        & $uv pip install --python $python torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
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

# Install with uv (much faster than pip)
& $uv pip install --python $python -r requirements_optimized.txt
& $uv pip install --python $python -r extra-req.txt

Remove-Item "requirements_optimized.txt" -Force -ErrorAction SilentlyContinue

# Cleanup caches to reduce package size
Write-Host "[INFO] Cleaning up caches..."
& $uv cache clean
Remove-Item "$env:USERPROFILE\.cache" -Recurse -Force -ErrorAction SilentlyContinue

# Remove unnecessary files from site-packages
$sitePackages = "$envPath\Lib\site-packages"

# Check if site-packages exists (uv structure)
if (-not (Test-Path $sitePackages)) {
    Write-Host "[INFO] site-packages not found at expected location, skipping cleanup"
} else {
    # Remove .pyc and .pyo files
    $pycFiles = Get-ChildItem $sitePackages -Recurse -Include "*.pyc", "*.pyo" -ErrorAction SilentlyContinue
    if ($pycFiles) {
        $pycFiles | Remove-Item -Force -ErrorAction SilentlyContinue
    }

    # Remove __pycache__ directories
    $pycacheDirs = Get-ChildItem $sitePackages -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
    if ($pycacheDirs) {
        $pycacheDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Remove test directories
    $testDirs = Get-ChildItem $sitePackages -Recurse -Directory -Filter "tests" -ErrorAction SilentlyContinue
    if ($testDirs) {
        $testDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    $testDir = Get-ChildItem $sitePackages -Recurse -Directory -Filter "test" -ErrorAction SilentlyContinue
    if ($testDir) {
        $testDir | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }

    # Remove Triton (not used on Windows)
    $tritonDirs = Get-ChildItem "$sitePackages" -Directory -Filter "triton*" -ErrorAction SilentlyContinue
    if ($tritonDirs) {
        $tritonDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Extract FFmpeg
Write-Host "[INFO] Extracting FFmpeg..."
Expand-Archive $ffZip -DestinationPath $tmpDir -Force
$ffDir = Get-ChildItem -Directory "$tmpDir" | Where-Object { $_.Name -like "ffmpeg*" } | Select-Object -First 1
Copy-Item "$($ffDir.FullName)\bin\ffmpeg.exe" "$envPath" -Force
Copy-Item "$($ffDir.FullName)\bin\ffprobe.exe" "$envPath" -Force
Remove-Item $ffDir.FullName -Recurse -Force -ErrorAction SilentlyContinue

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

# Download FunASR models using Python
Write-Host "[INFO] Downloading FunASR models..."
& $uv pip install --python $python "huggingface_hub[hf_xet]" -q

$asrModelsDir = "$srcDir\tools\asr\models"
New-Item -ItemType Directory -Force -Path $asrModelsDir | Out-Null

# Use environment variable or default
$HF_CACHE_REPO = if ($env:HF_MODELS_REPO) { $env:HF_MODELS_REPO } else { "sky1218/GPT-SoVITS-Models" }

# Create Python download script
$downloadScript = @"
import os
import sys
from pathlib import Path

def download_model(model_name, model_id, local_dir):
    """Download model from HuggingFace using huggingface_hub"""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"[INFO] Downloading {model_name} from HuggingFace...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            repo_type="model",
            allow_patterns=None,
            ignore_patterns=None,
            cache_dir=None,
            force_download=False,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        print(f"[INFO] Downloaded {model_name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to download {model_name}: {e}")
        return False

if __name__ == "__main__":
    model_name = sys.argv[1]
    model_id = sys.argv[2]
    local_dir = sys.argv[3]
    
    Path(local_dir).parent.mkdir(parents=True, exist_ok=True)
    download_model(model_name, model_id, local_dir)
"@

$downloadScript | Out-File "$tmpDir\download_model.py" -Encoding UTF8

# Download from single HuggingFace repo with subfolder
$downloadSubfolderScript = @"
import os
import sys
from pathlib import Path

def download_subfolder(repo_id, subfolder, local_dir):
    """Download subfolder from HuggingFace repo"""
    try:
        from huggingface_hub import snapshot_download
        
        print(f"[INFO] Downloading {subfolder} from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="model",
            allow_patterns=[f"{subfolder}/*"]
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
        
        print(f"[INFO] Downloaded {subfolder}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to download {subfolder}: {e}")
        return False

if __name__ == "__main__":
    repo_id = sys.argv[1]
    subfolder = sys.argv[2]
    local_dir = sys.argv[3]
    
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    download_subfolder(repo_id, subfolder, local_dir)
"@

$downloadSubfolderScript | Out-File "$tmpDir\download_subfolder.py" -Encoding UTF8

$funasr_models = @(
    "speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)

foreach ($modelName in $funasr_models) {
    $localDir = "$asrModelsDir\$modelName"
    
    if ((Test-Path $localDir) -and (Get-ChildItem $localDir -ErrorAction SilentlyContinue)) {
        Write-Host "[INFO] $modelName already exists, skipping"
        continue
    }
    
    # Use Python to download subfolder from HuggingFace
    & $python "$tmpDir\download_subfolder.py" $HF_CACHE_REPO $modelName $localDir
}

Write-Host "[INFO] All FunASR models downloaded successfully!"

# Clean up ModelScope cache files
$mscFiles = Get-ChildItem "$asrModelsDir" -Recurse -Filter ".msc" -ErrorAction SilentlyContinue
if ($mscFiles) {
    $mscFiles | Remove-Item -Force -ErrorAction SilentlyContinue
}

$lockFiles = Get-ChildItem "$asrModelsDir" -Recurse -Filter "*.lock" -ErrorAction SilentlyContinue
if ($lockFiles) {
    $lockFiles | Remove-Item -Force -ErrorAction SilentlyContinue
}

$gitFiles = Get-ChildItem "$asrModelsDir" -Recurse -Filter "*.git*" -ErrorAction SilentlyContinue
if ($gitFiles) {
    $gitFiles | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

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

$shFiles = Get-ChildItem "$srcDir" -Filter "*.sh" -ErrorAction SilentlyContinue
if ($shFiles) {
    $shFiles | Remove-Item -Force -ErrorAction SilentlyContinue
}

$ipynbFiles = Get-ChildItem "$srcDir" -Filter "*.ipynb" -ErrorAction SilentlyContinue
if ($ipynbFiles) {
    $ipynbFiles | Remove-Item -Force -ErrorAction SilentlyContinue
}

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
        Remove-Item "$junctionName" -Recurse -Force -ErrorAction SilentlyContinue
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
    Remove-Item $zstdZip -Force -ErrorAction SilentlyContinue
    Remove-Item "$zstdDir\zstd-v1.5.6-win64" -Recurse -Force -ErrorAction SilentlyContinue
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