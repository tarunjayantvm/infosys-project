#!/usr/bin/env python
"""
Download the pretrained UNet checkpoint from Google Drive.
This script runs during Render build before the app starts.
"""

import os
import sys

def download_checkpoint():
    """Download the checkpoint from Google Drive if it doesn't exist."""
    checkpoint_path = "pretrained_unet_checkpoint.pth"
    
    # If checkpoint already exists, skip download
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint already exists: {checkpoint_path}")
        return True
    
    try:
        import gdown
    except ImportError:
        print("ERROR: gdown is not installed. Install it with: pip install gdown")
        return False
    
    # ===== IMPORTANT: Replace this with YOUR Google Drive file ID =====
    # To get the file ID:
    # 1. Upload pretrained_unet_checkpoint.pth to your Google Drive
    # 2. Right-click the file → Share
    # 3. Copy the link (looks like: https://drive.google.com/file/d/YOUR_FILE_ID/view)
    # 4. Extract the ID part and paste it below
    
    file_id = "1b6Mb9awooNGRXTZJHz_4ExWuPbrKXpML"  # Your Google Drive file ID
    
    if file_id == "REPLACE_WITH_YOUR_GOOGLE_DRIVE_FILE_ID":
        print("ERROR: Please set your Google Drive file ID in download_model.py")
        print("Instructions in the file comments above.")
        return False
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"Downloading model checkpoint from Google Drive...")
    print(f"URL: {url}")
    
    try:
        gdown.download(url, checkpoint_path, quiet=False)
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"✓ Successfully downloaded {checkpoint_path} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"ERROR: File download failed - {checkpoint_path} not found")
            return False
    except Exception as e:
        print(f"ERROR during download: {e}")
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
