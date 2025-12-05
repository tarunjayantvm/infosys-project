from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import zipfile
import io
from datetime import datetime
import shutil
import numpy as np

import torch
import segmentation_models_pytorch as smp
import torchvision.transforms as T
from PIL import Image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['BATCH_FOLDER'] = 'static/batch_results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Increased to 50MB for batch
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['BATCH_FOLDER'], exist_ok=True)

# ----------------- MODEL LOADING (from checkpoint) -----------------
device = torch.device("cpu")  # or "cuda" if available

MODEL_PATH = "pretrained_unet_checkpoint.pth"
GDRIVE_FILE_ID = "1b6Mb9awooNGRXTZJHz_4ExWuPbrKXpML"  # Your file ID
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading model weights from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")
    
# 1) Same architecture as training
model = smp.Unet(
    encoder_name="resnet101",
    encoder_weights=None,   # using our own trained weights
    in_channels=3,
    classes=1,
    activation=None
)

# 2) Load checkpoint directly
checkpoint_path = "pretrained_unet_checkpoint.pth"  # put this file next to app.py
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# 3) Preprocessing exactly as in SegmentationPairedDataset
preprocess_224 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

to_tensor = T.ToTensor()

def predict_mask_from_pil(pil_img, threshold=0.3):
    """
    Returns preprocessed 224x224 image tensor and 224x224 binary mask.
    """
    img_t = preprocess_224(pil_img)                 # [3,224,224]
    x = img_t.unsqueeze(0).to(device)               # [1,3,224,224]

    with torch.no_grad():
        logits = model(x)                           # [1,1,224,224]
        probs = torch.sigmoid(logits)
        print("Sigmoid min/max:", float(probs.min()), float(probs.max()))
        mask = (probs > threshold).float()[0, 0].cpu()   # [224,224]

    return img_t, mask

# ----------------- FLASK ROUTES -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open original image
        pil_img = Image.open(filepath).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Run model prediction (224x224)
        img_t_224, mask_224 = predict_mask_from_pil(pil_img, threshold=0.3)

        # Resize mask back to original size
        mask_pil = T.ToPILImage()(mask_224)              # 224x224 -> PIL
        mask_full_pil = mask_pil.resize((orig_w, orig_h))
        mask_full_t = to_tensor(mask_full_pil)[0]        # [H,W] in {0,1}

        # Apply mask to ORIGINAL image
        img_orig_t = to_tensor(pil_img)                  # [3,H,W]
        masked_full = img_orig_t * mask_full_t.unsqueeze(0)  # [3,H,W]
        masked_full = masked_full.clamp(0.0, 1.0)

        masked_rgb = masked_full.permute(1, 2, 0).cpu().numpy()  # [H,W,3]

        # Save masked output image at original resolution
        base, ext = os.path.splitext(filename)
        masked_name = f"{base}_masked.png"
        masked_path = os.path.join(app.config['UPLOAD_FOLDER'], masked_name)
        Image.fromarray((masked_rgb * 255).astype('uint8')).save(masked_path)


        return jsonify({
            'success': True,
            'filename': filename,
            'url_original': f'/static/uploads/{filename}',
            'url_masked': f'/static/uploads/{masked_name}'
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/batch-upload', methods=['POST'])
def batch_upload():
    """Handle multiple file uploads and return a ZIP with all results"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')
    
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400

    # Validate files
    valid_files = []
    errors = []
    for file in files:
        if file.filename == '':
            continue
        if not allowed_file(file.filename):
            errors.append(f"{file.filename}: Invalid file type")
            continue
        valid_files.append(file)

    if not valid_files:
        return jsonify({'error': 'No valid image files found. Supported: PNG, JPG, JPEG, GIF'}), 400

    # Create batch folder with timestamp
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(app.config['BATCH_FOLDER'], f"temp_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)

    results = {
        'batch_id': batch_id,
        'total_files': len(valid_files),
        'processed_files': 0,
        'files': []
    }

    # Process each image
    for file in valid_files:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Open original image
            pil_img = Image.open(filepath).convert("RGB")
            orig_w, orig_h = pil_img.size

            # Run model prediction
            img_t_224, mask_224 = predict_mask_from_pil(pil_img, threshold=0.3)

            # Resize mask back to original size
            mask_pil = T.ToPILImage()(mask_224)
            mask_full_pil = mask_pil.resize((orig_w, orig_h))
            mask_full_t = to_tensor(mask_full_pil)[0]

            # Apply mask to ORIGINAL image
            img_orig_t = to_tensor(pil_img)
            masked_full = img_orig_t * mask_full_t.unsqueeze(0)
            masked_full = masked_full.clamp(0.0, 1.0)

            masked_rgb = masked_full.permute(1, 2, 0).cpu().numpy()

            # Save to batch folder
            base, ext = os.path.splitext(filename)
            masked_name = f"{base}_segmented.png"
            masked_path = os.path.join(batch_dir, masked_name)
            Image.fromarray((masked_rgb * 255).astype('uint8')).save(masked_path)


            # Also save original to batch folder
            orig_batch_path = os.path.join(batch_dir, filename)
            pil_img.save(orig_batch_path)

            results['files'].append({
                'original': filename,
                'segmented': masked_name,
                'size': f"{orig_w}x{orig_h}"
            })
            results['processed_files'] += 1

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    # Create ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all files from batch folder
        for file in os.listdir(batch_dir):
            file_path = os.path.join(batch_dir, file)
            arcname = os.path.join(f"AI_Vision_Batch_{batch_id}", file)
            zip_file.write(file_path, arcname)

    zip_buffer.seek(0)

    # Save ZIP file also on server for later download
    zip_filename = f"batch_results_{batch_id}.zip"
    zip_path = os.path.join(app.config['BATCH_FOLDER'], zip_filename)
    with open(zip_path, 'wb') as f:
        f.write(zip_buffer.getvalue())

    # Keep batch folder for gallery display (don't delete)
    # It will be cleaned up periodically or manually

    return jsonify({
        'success': True,
        'batch_id': batch_id,
        'processed_files': results['processed_files'],
        'total_files': results['total_files'],
        'download_url': f'/download-batch/{batch_id}',
        'files': results['files'],
        'errors': errors
    })

def _render_masked_bytes(pil_img, fmt='png'):
    """Render masked output bytes for given PIL image and requested format.
    fmt: 'png', 'png_transparent', 'jpeg', 'webp', 'mask'
    Returns (BytesIO_buffer, extension)
    """
    pil_img = pil_img.convert('RGB')
    orig_w, orig_h = pil_img.size

    # Recompute mask to ensure accurate segmentation
    _, mask_224 = predict_mask_from_pil(pil_img, threshold=0.3)
    mask_pil = T.ToPILImage()(mask_224)
    mask_full_pil = mask_pil.resize((orig_w, orig_h)).convert('L')
    mask_arr = np.array(mask_full_pil)

    orig_arr = np.array(pil_img)

    buf = io.BytesIO()

    if fmt in ('png', 'jpeg', 'webp'):
        mask_bool = mask_arr > 128
        masked = orig_arr.copy()
        masked[~mask_bool] = 0
        out = Image.fromarray(masked)
        pil_fmt = 'PNG' if fmt == 'png' else ('JPEG' if fmt == 'jpeg' else 'WEBP')
        out.save(buf, format=pil_fmt, quality=95)
        ext = 'png' if fmt == 'png' else ('jpg' if fmt == 'jpeg' else 'webp')

    elif fmt == 'png_transparent':
        alpha = (mask_arr > 128).astype('uint8') * 255
        rgba = np.dstack((orig_arr, alpha))
        out = Image.fromarray(rgba, mode='RGBA')
        out.save(buf, format='PNG')
        ext = 'png'

    elif fmt == 'mask':
        # Save mask as single-channel PNG
        out = Image.fromarray((mask_arr).astype('uint8'))
        out.save(buf, format='PNG')
        ext = 'png'

    else:
        # default PNG
        mask_bool = mask_arr > 128
        masked = orig_arr.copy()
        masked[~mask_bool] = 0
        out = Image.fromarray(masked)
        out.save(buf, format='PNG')
        ext = 'png'

    buf.seek(0)
    return buf, ext


@app.route('/download-result')
def download_result():
    """Download a single processed result in requested format.
    Query params: filename, format
    formats: png, png_transparent, jpeg, webp, mask
    """
    filename = request.args.get('filename')
    fmt = request.args.get('format', 'png')
    if not filename:
        return jsonify({'error': 'filename required'}), 400

    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if not os.path.exists(orig_path):
        return jsonify({'error': 'Original image not found'}), 404

    pil_img = Image.open(orig_path).convert('RGB')
    buf, ext = _render_masked_bytes(pil_img, fmt)

    download_name = f"{os.path.splitext(filename)[0]}_segmented.{ext}"
    return send_file(buf, mimetype='application/octet-stream', as_attachment=True, download_name=download_name)


@app.route('/download-batch-zip/<batch_id>')
def download_batch_zip(batch_id):
    """Generate ZIP for a batch in the requested format.
    Query param: format (png, png_transparent, jpeg, webp, mask)
    """
    fmt = request.args.get('format', 'png')
    batch_dir = os.path.join(app.config['BATCH_FOLDER'], f"temp_{batch_id}")
    if not os.path.exists(batch_dir):
        return jsonify({'error': 'Batch not found'}), 404

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fname in os.listdir(batch_dir):
            low = fname.lower()
            if any(low.endswith(ext) for ext in app.config['ALLOWED_EXTENSIONS']):
                orig_path = os.path.join(batch_dir, fname)
                try:
                    pil_img = Image.open(orig_path).convert('RGB')
                    buf, ext = _render_masked_bytes(pil_img, fmt)
                    arcname = f"{os.path.splitext(fname)[0]}_segmented.{ext}"
                    zipf.writestr(arcname, buf.getvalue())
                except Exception as e:
                    print(f"Error rendering {fname}: {e}")
                    continue

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name=f"AI_Vision_Results_{batch_id}.zip")


@app.route('/download-batch/<batch_id>')
def download_batch(batch_id):
    """Download batch results as ZIP (pre-generated)."""
    zip_filename = f"batch_results_{batch_id}.zip"
    zip_path = os.path.join(app.config['BATCH_FOLDER'], zip_filename)
    
    if not os.path.exists(zip_path):
        return jsonify({'error': 'Batch not found'}), 404
    
    return send_file(
        zip_path,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"AI_Vision_Results_{batch_id}.zip"
    )


@app.route('/batch-image/<batch_id>/<filename>')
def get_batch_image(batch_id, filename):
    """Serve batch images for display"""
    # Extract batch folder from zip if needed, or serve from temp storage
    batch_dir = os.path.join(app.config['BATCH_FOLDER'], f"temp_{batch_id}")
    file_path = os.path.join(batch_dir, filename)
    
    if os.path.exists(file_path):
        from flask import send_from_directory
        return send_from_directory(batch_dir, filename)
    
    return jsonify({'error': 'Image not found'}), 404


@app.route('/editor')
def editor():
    """Client-side image editor page."""
    return render_template('editor.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
