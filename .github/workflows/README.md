# GitHub Actions Workflows

This document describes the available workflows and required setup.

## Workflows Overview

| Workflow | Description | Trigger |
|----------|-------------|---------|
| `build_windows_packages.yaml` | Build Windows packages and upload to HuggingFace/ModelScope | Manual |
| `cache_funasr_models.yaml` | Cache FunASR models from ModelScope to HuggingFace | Manual |
| `docker-publish.yaml` | Build and publish Docker image | Manual/Push |

## Required Secrets

Go to **Settings → Secrets and variables → Actions** to add these secrets:

| Secret | Required By | Description | How to Get |
|--------|-------------|-------------|------------|
| `HF_TOKEN` | All workflows | HuggingFace API token | [HuggingFace Settings](https://huggingface.co/settings/tokens) → New token (Write access) |
| `MS_TOKEN` | `build_windows_packages.yaml` | ModelScope API token | [ModelScope Settings](https://www.modelscope.cn/my/myaccesstoken) → Create token |

## Workflow Details

### 1. Build Windows Packages

**File:** `build_windows_packages.yaml`

Builds Windows packages with CUDA support and uploads to HuggingFace and ModelScope.

**Inputs:**
- `date` (optional): Date suffix for package name (default: current date MMDD)
- `suffix` (optional): Additional suffix for package name

**What it does:**
1. Creates Python environment with micromamba
2. Installs PyTorch with CUDA (cu126 and cu128)
3. Downloads pretrained models
4. Packages everything into `.tar.zst`
5. Uploads to HuggingFace and ModelScope
6. Creates GitHub Release

**Run:**
```bash
gh workflow run build_windows_packages.yaml
```

### 2. Cache FunASR Models

**File:** `cache_funasr_models.yaml`

Downloads FunASR models from ModelScope and caches them to HuggingFace for faster access.

**Inputs:**
- `hf_repo` (optional): Target HuggingFace repo (default: `sky1218/GPT-SoVITS-Models`)

**Run:**
```bash
gh workflow run cache_funasr_models.yaml
```

### 3. Docker Publish

**File:** `docker-publish.yaml`

Builds and publishes Docker image to GitHub Container Registry.

**Run:**
```bash
gh workflow run docker-publish.yaml
```

## Quick Setup

1. Fork or clone this repository
2. Add required secrets (see table above)
3. Run workflows from Actions tab or using GitHub CLI

```bash
# Set secrets
gh secret set HF_TOKEN --body "hf_xxxxx"
gh secret set MS_TOKEN --body "ms-xxxxx"

# Run build
gh workflow run build_windows_packages.yaml
```
