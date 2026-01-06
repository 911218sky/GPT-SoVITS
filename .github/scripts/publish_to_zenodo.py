#!/usr/bin/env python3
"""
Update Zenodo deposition metadata and publish.
"""
import os
import sys
import json
import requests


def update_and_publish_zenodo(deposition_id: str, token: str, version: str):
    """
    Update Zenodo deposition metadata and publish it.
    
    Args:
        deposition_id: Zenodo deposition ID
        token: Zenodo API token
        version: Package version (e.g., v0106)
    """
    base_url = "https://zenodo.org/api/deposit/depositions"
    
    # Prepare metadata
    metadata = {
        "metadata": {
            "title": f"GPT-SoVITS Windows Package {version}",
            "upload_type": "software",
            "description": (
                "GPT-SoVITS Windows Package with Python 3.11 + PyTorch. "
                "This package includes both CUDA 12.4 and CUDA 12.8 versions. "
                "Contains all pretrained models (v2Pro, sv, G2PW), FunASR models, and FFmpeg."
            ),
            "creators": [
                {"name": "GPT-SoVITS Team"}
            ],
            "keywords": [
                "GPT-SoVITS",
                "TTS",
                "Text-to-Speech",
                "Voice Cloning",
                "AI",
                "PyTorch",
                "CUDA"
            ],
            "license": "MIT",
            "version": version
        }
    }
    
    print("Updating deposition metadata...")
    
    # Update metadata
    try:
        response = requests.put(
            f"{base_url}/{deposition_id}",
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            },
            json=metadata,
            timeout=30
        )
        
        if response.status_code not in [200, 201]:
            print(f"✗ Failed to update metadata: {response.status_code}")
            print(response.text)
            sys.exit(1)
        
        print("✓ Metadata updated")
        
    except Exception as e:
        print(f"✗ Error updating metadata: {str(e)}")
        sys.exit(1)
    
    # Publish the deposition
    print("Publishing to Zenodo...")
    
    try:
        response = requests.post(
            f"{base_url}/{deposition_id}/actions/publish",
            headers={
                'Authorization': f'Bearer {token}'
            },
            timeout=30
        )
        
        if response.status_code not in [200, 201, 202]:
            print(f"✗ Failed to publish: {response.status_code}")
            print(response.text)
            sys.exit(1)
        
        result = response.json()
        doi = result.get('doi', 'N/A')
        record_id = result.get('id', 'N/A')
        zenodo_url = f"https://zenodo.org/records/{record_id}"
        
        print("✓ Published to Zenodo!")
        print(f"DOI: {doi}")
        print(f"URL: {zenodo_url}")
        
        # Output for GitHub Actions
        if 'GITHUB_OUTPUT' in os.environ:
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"doi={doi}\n")
                f.write(f"url={zenodo_url}\n")
                f.write(f"record_id={record_id}\n")
        
    except Exception as e:
        print(f"✗ Error publishing: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point"""
    deposition_id = os.environ.get('DEPOSITION_ID')
    token = os.environ.get('ZENODO_TOKEN')
    version = os.environ.get('VERSION')
    
    if not deposition_id:
        print("✗ Error: DEPOSITION_ID environment variable not set")
        sys.exit(1)
    
    if not token:
        print("✗ Error: ZENODO_TOKEN environment variable not set")
        sys.exit(1)
    
    if not version:
        print("✗ Error: VERSION environment variable not set")
        sys.exit(1)
    
    update_and_publish_zenodo(deposition_id, token, version)


if __name__ == '__main__':
    main()
