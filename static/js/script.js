document.addEventListener('DOMContentLoaded', function() {
    // ==================== ELEMENTS ====================
    const modeRadios = document.querySelectorAll('input[name="mode"]');
    const singleUploadSection = document.getElementById('singleUploadSection');
    const batchUploadSection = document.getElementById('batchUploadSection');
    
    const tryItBtn = document.getElementById('tryItBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const singleUploadCard = document.getElementById('singleUploadCard');
    
    const batchUploadBtn = document.getElementById('batchUploadBtn');
    const batchFileInput = document.getElementById('batchFileInput');
    const batchUploadCard = document.getElementById('batchUploadCard');
    const fileList = document.getElementById('fileList');
    const fileListItems = document.getElementById('fileListItems');
    
    const resultSection = document.getElementById('resultSection');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');
    const singleResult = document.getElementById('singleResult');
    const batchResult = document.getElementById('batchResult');
    
    const uploadedImageWrapper = document.getElementById('uploadedImageWrapper');
    const maskedImageWrapper = document.getElementById('maskedImageWrapper');
    const downloadBtn = document.getElementById('downloadBtn');
    const tryAnotherBtn = document.getElementById('tryAnotherBtn');
    
    const downloadBatchBtn = document.getElementById('downloadBatchBtn');
    const tryBatchAgainBtn = document.getElementById('tryBatchAgainBtn');
    
    const loadingIndicator = document.getElementById('loadingIndicator');

    let currentMode = 'single';
    let selectedFiles = [];
    let currentResultData = null;
    let currentBatchData = null;

    // ==================== MODE SWITCHING ====================
    modeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentMode = e.target.value;
            if (currentMode === 'single') {
                singleUploadSection.style.display = 'block';
                batchUploadSection.style.display = 'none';
            } else {
                singleUploadSection.style.display = 'none';
                batchUploadSection.style.display = 'block';
            }
        });
    });

    // ==================== SINGLE UPLOAD ====================
    tryItBtn.addEventListener('click', () => fileInput.click());
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) uploadImage(file);
    });

    // Single upload drag & drop
    setupDragDrop(singleUploadCard, (file) => {
        if (file.type.startsWith('image/')) uploadImage(file);
        else showAlert('Please drop a valid image file', 'error');
    });

    // ==================== BATCH UPLOAD ====================
    batchUploadBtn.addEventListener('click', () => batchFileInput.click());

    batchFileInput.addEventListener('change', (e) => {
        selectedFiles = Array.from(e.target.files);
        updateFileList();
    });

    // Batch upload drag & drop
    setupDragDrop(batchUploadCard, (file) => {
        if (file.type.startsWith('image/')) {
            selectedFiles.push(file);
            updateFileList();
        }
    });

    // ==================== RESULT BUTTONS ====================
    downloadBtn.addEventListener('click', () => {
        if (currentResultData) downloadSingleResult(currentResultData);
    });

    tryAnotherBtn.addEventListener('click', () => {
        resetForm();
        fileInput.click();
    });

    downloadBatchBtn.addEventListener('click', () => {
        if (currentBatchData) downloadBatchZip(currentBatchData);
    });

    tryBatchAgainBtn.addEventListener('click', () => {
        selectedFiles = [];
        fileListItems.innerHTML = '';
        fileList.style.display = 'none';
        batchFileInput.value = '';
        resultSection.style.display = 'none';
        batchFileInput.click();
    });

    // ==================== UPLOAD FUNCTIONS ====================
    function uploadImage(file) {
        if (!file.type.startsWith('image/')) {
            showAlert('Please select a valid image file', 'error');
            return;
        }

        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            showAlert('File size exceeds 16MB limit', 'error');
            return;
        }

        loadingIndicator.style.display = 'flex';
        resultSection.style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            if (data.success) {
                currentResultData = data;
                displaySingleResult(data);
                singleResult.style.display = 'block';
                batchResult.style.display = 'none';
                resultTitle.textContent = 'Processing Complete!';
                resultSubtitle.textContent = 'Your result is ready';
                resultSection.style.display = 'block';
                setTimeout(() => resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' }), 300);
            } else {
                showAlert(data.error || 'Error processing image', 'error');
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            console.error('Error:', error);
            showAlert('Error uploading image. Please try again.', 'error');
        });
    }

    function uploadBatch() {
        if (selectedFiles.length === 0) {
            showAlert('Please select at least one image', 'error');
            return;
        }

        loadingIndicator.style.display = 'flex';
        resultSection.style.display = 'none';

        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files[]', file);
        });

        fetch('/batch-upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            if (data.success) {
                currentBatchData = data;
                displayBatchResult(data);
                singleResult.style.display = 'none';
                batchResult.style.display = 'block';
                resultTitle.textContent = 'Batch Processing Complete!';
                resultSubtitle.textContent = `Successfully processed ${data.processed_files} out of ${data.total_files} images`;
                resultSection.style.display = 'block';
                setTimeout(() => resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' }), 300);
            } else {
                showAlert(data.error || 'Error processing batch', 'error');
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            console.error('Error:', error);
            showAlert('Error uploading batch. Please try again.', 'error');
        });
    }

    // ==================== DISPLAY RESULTS ====================
    function displaySingleResult(data) {
        uploadedImageWrapper.innerHTML = '';
        const uploadedImg = document.createElement('img');
        uploadedImg.src = data.url_original;
        uploadedImg.alt = 'Uploaded Image';
        uploadedImg.style.width = '100%';
        uploadedImg.style.height = '100%';
        uploadedImg.style.objectFit = 'contain';
        uploadedImageWrapper.appendChild(uploadedImg);

        maskedImageWrapper.innerHTML = '';
        const maskedImg = document.createElement('img');
        maskedImg.src = data.url_masked;
        maskedImg.alt = 'Masked Output';
        maskedImg.style.width = '100%';
        maskedImg.style.height = '100%';
        maskedImg.style.objectFit = 'contain';
        maskedImageWrapper.appendChild(maskedImg);
    }

    function displayBatchResult(data) {
        document.getElementById('statTotal').textContent = data.total_files;
        document.getElementById('statProcessed').textContent = data.processed_files;
        document.getElementById('statErrors').textContent = data.total_files - data.processed_files;

        const batchGallery = document.getElementById('batchGallery');
        batchGallery.innerHTML = '';

        data.files.forEach((file, index) => {
            const galleryItem = document.createElement('div');
            galleryItem.className = 'gallery-item';
            galleryItem.innerHTML = `
                <div class="gallery-item-title">
                    <i class="fas fa-images me-2"></i>Image ${index + 1} - ${file.original}
                </div>
                
                <div class="gallery-compare">
                    <div class="gallery-compare-label">
                        <i class="fas fa-image me-1"></i>Original Image
                    </div>
                    <div class="gallery-compare-image">
                        <img src="/batch-image/${data.batch_id}/${file.original}" alt="Original">
                    </div>
                </div>
                
                <div class="gallery-compare">
                    <div class="gallery-compare-label">
                        <i class="fas fa-magic me-1"></i>Segmented Output
                    </div>
                    <div class="gallery-compare-image">
                        <img src="/batch-image/${data.batch_id}/${file.segmented}" alt="Segmented">
                    </div>
                </div>
            `;
            batchGallery.appendChild(galleryItem);
        });
    }

    // ==================== DOWNLOAD FUNCTIONS ====================
    function downloadSingleResult(data) {
        const formatSelect = document.getElementById('formatSelectSingle');
        const fmt = formatSelect ? formatSelect.value : 'png';
        const url = `/download-result?filename=${encodeURIComponent(data.filename)}&format=${encodeURIComponent(fmt)}`;
        const link = document.createElement('a');
        link.href = url;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function downloadBatchZip(data) {
        const formatSelect = document.getElementById('formatSelectBatch');
        const fmt = formatSelect ? formatSelect.value : 'png';
        const url = `/download-batch-zip/${data.batch_id}?format=${encodeURIComponent(fmt)}`;
        const link = document.createElement('a');
        link.href = url;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // ==================== FILE LIST MANAGEMENT ====================
    function updateFileList() {
        if (selectedFiles.length === 0) {
            fileList.style.display = 'none';
            return;
        }

        fileList.style.display = 'block';
        fileListItems.innerHTML = '';

        selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <div>
                    <i class="fas fa-image"></i>
                    <span>${file.name}</span>
                    <span style="color: var(--text-muted); font-size: 0.8rem; margin-left: 0.5rem;">(${(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                </div>
                <button class="file-item-remove" onclick="removeFile(${index})">Remove</button>
            `;
            fileListItems.appendChild(fileItem);
        });

        // Add upload button
        const uploadBatchSection = document.querySelector('.upload-section');
        let batchButtonContainer = document.getElementById('batchButtonContainer');
        if (!batchButtonContainer) {
            batchButtonContainer = document.createElement('div');
            batchButtonContainer.id = 'batchButtonContainer';
            batchButtonContainer.style.marginTop = '1.5rem';
            batchButtonContainer.style.textAlign = 'center';
            batchButtonContainer.innerHTML = `
                <button class="btn btn-success btn-lg fw-bold">
                    <i class="fas fa-check me-2"></i>Upload ${selectedFiles.length} Image(s)
                </button>
            `;
            batchButtonContainer.addEventListener('click', uploadBatch);
            const fileListParent = fileList.parentElement;
            fileListParent.appendChild(batchButtonContainer);
        }
    }

    window.removeFile = function(index) {
        selectedFiles.splice(index, 1);
        updateFileList();
    };

    // ==================== HELPER FUNCTIONS ====================
    function setupDragDrop(element, callback) {
        element.addEventListener('dragover', (e) => {
            e.preventDefault();
            element.classList.add('drag-over');
        });

        element.addEventListener('dragleave', () => {
            element.classList.remove('drag-over');
        });

        element.addEventListener('drop', (e) => {
            e.preventDefault();
            element.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                if (currentMode === 'batch') {
                    selectedFiles.push(...Array.from(files));
                    updateFileList();
                } else {
                    callback(files[0]);
                }
            }
        });
    }

    function resetForm() {
        fileInput.value = '';
        resultSection.style.display = 'none';
        uploadedImageWrapper.innerHTML = '<div class="placeholder"><i class="fas fa-image"></i></div>';
        maskedImageWrapper.innerHTML = '<div class="placeholder"><i class="fas fa-magic"></i></div>';
    }

    function showAlert(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : 'info'} alert-dismissible fade show`;
        alertDiv.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 9998;
            max-width: 400px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        `;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        setTimeout(() => alertDiv.remove(), 5000);
    }

    // Navigation smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth' });
        });
    });
});
