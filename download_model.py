#!/usr/bin/env python
"""
Download the pretrained UNet checkpoint from Google Drive.
This script runs during Render build before the app starts.
"""

import os
import sys
import re
try:
    import requests
except Exception:
    requests = None

def download_checkpoint():
    """Download the checkpoint from Google Drive if it doesn't exist."""
    checkpoint_path = "pretrained_unet_checkpoint.pth"
    
    # If checkpoint already exists, skip download
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint already exists: {checkpoint_path}")
        return True
    
    # Prefer environment variables for flexible deployments
    download_url = os.getenv("DOWNLOAD_URL")
    env_file_id = os.getenv("GDRIVE_FILE_ID")

    # Helper: save streaming response to file
    def _save_response(resp, dest_path):
        try:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"ERROR saving response: {e}")
            return False

    # Helper: attempt to download a direct URL using requests
    def _download_from_url(url, dest):
        if not requests:
            print("requests not available in environment; falling back to gdown if present")
            return False
        try:
            print(f"Attempting direct download from URL: {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                return _save_response(r, dest)
        except Exception as e:
            print(f"Direct download failed: {e}")
            return False

    # Helper: Google Drive large-file download flow using requests
    def _download_from_gdrive(file_id, dest):
        if not requests:
            print("requests not available; falling back to gdown if present")
            return False
        session = requests.Session()
        URL = "https://docs.google.com/uc?export=download"
        try:
            resp = session.get(URL, params={"id": file_id}, stream=True, timeout=60)
        except Exception as e:
            print(f"Initial request to Google Drive failed: {e}")
            return False

        # Check for confirmation token in cookies
        token = None
        for k, v in resp.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

        # Fallback: parse token from HTML
        if not token:
            try:
                m = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text)
                if m:
                    token = m.group(1)
            except Exception:
                token = None

        if token:
            try:
                resp = session.get(URL, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
            except Exception as e:
                print(f"Confirmed request failed: {e}")
                return False

        return _save_response(resp, dest)

    # 1) Try explicit DOWNLOAD_URL env var (direct host or drive link)
    if download_url:
        print(f"DOWNLOAD_URL provided: {download_url}")
        # If it's a Google Drive 'uc' link, try the requests-drive flow
        if "drive.google.com" in download_url or "docs.google.com" in download_url:
            # try requests direct first
            ok = _download_from_url(download_url, checkpoint_path)
            if ok:
                size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"✓ Successfully downloaded {checkpoint_path} ({size_mb:.2f} MB)")
                return True
            # try gdrive flow by extracting id if possible
            id_match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", download_url)
            if id_match:
                file_id = id_match.group(1)
                if _download_from_gdrive(file_id, checkpoint_path):
                    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                    print(f"✓ Successfully downloaded {checkpoint_path} ({size_mb:.2f} MB)")
                    return True
        else:
            if _download_from_url(download_url, checkpoint_path):
                size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                print(f"✓ Successfully downloaded {checkpoint_path} ({size_mb:.2f} MB)")
                return True

    # 2) Try environment GDRIVE_FILE_ID
    if env_file_id:
        print(f"GDRIVE_FILE_ID provided via env: {env_file_id}")
        if _download_from_gdrive(env_file_id, checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✓ Successfully downloaded {checkpoint_path} ({size_mb:.2f} MB)")
            return True

    # 3) Fall back to hard-coded ID in this file (legacy behavior)
    file_id = "1b6Mb9awooNGRXTZJHz_4ExWuPbrKXpML"
    print(f"Falling back to hard-coded Google Drive ID: {file_id}")
    if _download_from_gdrive(file_id, checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"✓ Successfully downloaded {checkpoint_path} ({size_mb:.2f} MB)")
        return True

    # 4) Last-resort: try gdown if available (older behavior)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Attempting download with gdown (fallback): {url}")
        gdown.download(url, checkpoint_path, quiet=False)
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✓ Successfully downloaded {checkpoint_path} ({size_mb:.2f} MB)")
            return True
    except Exception as e:
        print(f"gdown fallback failed or not available: {e}")

    # If we reach here, download failed. Honor FORCE_START_ON_DOWNLOAD_FAIL if set.
    force_flag = os.getenv("FORCE_START_ON_DOWNLOAD_FAIL", "0")
    if force_flag == "1":
        print("WARNING: Model download failed but FORCE_START_ON_DOWNLOAD_FAIL=1, continuing without model.")
        return True

    return False

if __name__ == "__main__":
    print("=" * 60)
    print("AI Vision Extract - Model Download Script")
    print("=" * 60)
    
    success = download_checkpoint()
    
    if success:
        print("\n✓ Model download completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Model download failed. Please check the error above.")
        sys.exit(1)
