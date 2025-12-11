# Option 2: Google Drive Checkpoint Download - Setup Guide

## What's Been Done

✅ Created `download_model.py` - script to download checkpoint from Google Drive  
✅ Updated `Procfile` - now runs download script before starting Flask  
✅ Updated `.gitignore` - excludes `*.pth` files from GitHub  
✅ `requirements.txt` - already includes `gdown` and `gunicorn`  

---

## Step 1: Upload Your Checkpoint to Google Drive

1. Open **[Google Drive](https://drive.google.com)**
2. Create a new folder (e.g., `AI Vision Models`)
3. Upload your `pretrained_unet_checkpoint.pth` to that folder
4. Right-click the file → **Share**
5. Change permission to **"Anyone with the link"**
6. Copy the link (looks like):
   ```
   https://drive.google.com/file/d/1abc123XYZ_defGHI456jk/view
   ```

---

## Step 2: Extract the Google Drive File ID

From the link above, extract the ID between `/d/` and `/`:
```
https://drive.google.com/file/d/ [THIS_PART] /view
                                  ↑
                            FILE_ID = 1abc123XYZ_defGHI456jk
```

---

## Step 3: Update `download_model.py`

Open `download_model.py` and find this line (around line 33):

```python
file_id = "REPLACE_WITH_YOUR_GOOGLE_DRIVE_FILE_ID"  # <-- CHANGE THIS
```

Replace it with your actual file ID:

```python
file_id = "1abc123XYZ_defGHI456jk"  # Replace with YOUR file ID
```

**Save the file.**

---

## Step 4: Test Locally (Optional)

Make sure the script works before deploying:

```powershell
cd c:\Users\Tarun\Downloads\AIVisionExtract\project

# Delete the local checkpoint to test download
Remove-Item pretrained_unet_checkpoint.pth -Force

# Test the download script
python download_model.py

# Should output:
# ✓ Successfully downloaded pretrained_unet_checkpoint.pth (590.45 MB)
```

---

## Step 5: Push to GitHub

```powershell
cd c:\Users\Tarun\Downloads\AIVisionExtract\project

git add .
git commit -m "Setup Option 2: Download checkpoint from Google Drive during build"
git push origin main
```

**Verify on GitHub:**
- `download_model.py` ✓ (new file)
- `Procfile` ✓ (updated)
- `.gitignore` ✓ (updated with *.pth)
- `pretrained_unet_checkpoint.pth` ✗ (should NOT be present - ignored)

---

## Step 6: Deploy on Render

1. Go to **[render.com](https://render.com)**
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Fill in the details:
   - **Name:** `ai-vision-extract`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Python Version:** `3.10.14`

5. Click **"Create Web Service"**

**What happens:**
- Render clones your GitHub repo
- Installs requirements from `requirements.txt`
- Runs `Procfile`: `python download_model.py && gunicorn app:app`
- `download_model.py` downloads the checkpoint from Google Drive
- Flask app starts with the model loaded

---

## ✅ Checklist

- [ ] Uploaded checkpoint to Google Drive
- [ ] Made Google Drive file shareable (link access enabled)
- [ ] Extracted the file ID
- [ ] Updated `download_model.py` with your file ID
- [ ] Tested locally: `python download_model.py` works
- [ ] Committed and pushed to GitHub
- [ ] Verified checkpoint is NOT in GitHub repo (checked `.gitignore`)
- [ ] Deployed on Render

---

## Troubleshooting

### "ERROR: Please set your Google Drive file ID"
→ You didn't update the `file_id` variable in `download_model.py`

### "ERROR: File download failed"
→ Check that your Google Drive link is public and the file exists

### Render deployment fails with "File not found" error
→ The download script didn't run; check Render logs at the bottom of your dashboard

---

## Questions?

If something goes wrong during deployment, check Render's **Logs** tab for errors and send me the error message!

Good luck! 🚀
