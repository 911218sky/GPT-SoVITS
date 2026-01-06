#!/usr/bin/env python3
"""
Upload packages to Zenodo with progress bar.
"""
import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm


def upload_files_to_zenodo(bucket_url: str, token: str, packages_dir: str = 'packages'):
    """
    Upload all .7z files in the packages directory to Zenodo.
    
    Args:
        bucket_url: Zenodo bucket URL from deposition creation
        token: Zenodo API token
        packages_dir: Directory containing .7z files to upload
    """
    packages_path = Path(packages_dir)
    
    if not packages_path.exists():
        print(f"✗ Error: Directory '{packages_dir}' not found")
        sys.exit(1)
    
    files = list(packages_path.glob('*.7z'))
    
    if not files:
        print(f"✗ Error: No .7z files found in '{packages_dir}'")
        sys.exit(1)
    
    print(f"Found {len(files)} file(s) to upload\n")
    
    for file_path in files:
        filename = file_path.name
        filesize = file_path.stat().st_size
        
        print(f"Uploading {filename} ({filesize:,} bytes) to Zenodo...")
        
        # Create a progress bar
        with tqdm(
            total=filesize,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=filename
        ) as pbar:
            
            # Create a wrapper to update progress bar
            def read_in_chunks(file_object, chunk_size=1024*1024):  # 1MB chunks
                """Read file in chunks and update progress bar"""
                while True:
                    data = file_object.read(chunk_size)
                    if not data:
                        break
                    pbar.update(len(data))
                    yield data
            
            # Upload with streaming
            try:
                with open(file_path, 'rb') as f:
                    response = requests.put(
                        f"{bucket_url}/{filename}",
                        data=read_in_chunks(f),
                        headers={
                            'Authorization': f'Bearer {token}',
                            'Content-Type': 'application/octet-stream'
                        },
                        timeout=3600  # 1 hour timeout for large files
                    )
                
                if response.status_code in [200, 201]:
                    print(f"✓ Uploaded {filename}\n")
                else:
                    print(f"✗ Failed to upload {filename}: {response.status_code}")
                    print(response.text)
                    sys.exit(1)
                    
            except Exception as e:
                print(f"✗ Error uploading {filename}: {str(e)}")
                sys.exit(1)
    
    print("All files uploaded successfully!")


def main():
    """Main entry point"""
    bucket_url = os.environ.get('BUCKET_URL')
    token = os.environ.get('ZENODO_TOKEN')
    packages_dir = os.environ.get('PACKAGES_DIR', 'packages')
    
    if not bucket_url:
        print("✗ Error: BUCKET_URL environment variable not set")
        sys.exit(1)
    
    if not token:
        print("✗ Error: ZENODO_TOKEN environment variable not set")
        sys.exit(1)
    
    upload_files_to_zenodo(bucket_url, token, packages_dir)


if __name__ == '__main__':
    main()
