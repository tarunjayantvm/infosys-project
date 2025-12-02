document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('editorFileInput');
    const canvas = document.getElementById('editorCanvas');
    const ctx = canvas.getContext('2d');
    const undoBtn = document.getElementById('undoBtn');
    const resetBtn = document.getElementById('resetBtn');
    const exportBtn = document.getElementById('exportBtn');

    // tools
    const toolButtons = {
        select: document.getElementById('tool-select'),
        lasso: document.getElementById('tool-lasso'),
        rotate: document.getElementById('tool-rotate'),
        flipH: document.getElementById('tool-flip-h'),
        flipV: document.getElementById('tool-flip-v'),
        resize: document.getElementById('tool-resize')
    };

    const options = {
        crop: document.getElementById('options-crop'),
        resize: document.getElementById('options-resize'),
        lasso: document.getElementById('options-lasso'),
        default: document.getElementById('options-default')
    };

    let img = new Image();
    let originalImage = null;
    let history = [];

    let currentTool = null;

    function setActiveTool(name) {
        currentTool = name;
        Object.values(toolButtons).forEach(btn => btn.classList.remove('active'));
        Object.keys(options).forEach(k => options[k].style.display = 'none');
        options.default.style.display = 'none';
        if (!name) { options.default.style.display = 'block'; return; }
        if (name === 'select') { toolButtons.select.classList.add('active'); options.crop.style.display = 'block'; }
        if (name === 'lasso') { toolButtons.lasso.classList.add('active'); options.lasso.style.display = 'block'; }
        if (name === 'resize') { toolButtons.resize.classList.add('active'); options.resize.style.display = 'block'; }
    }

    // draw helpers
    function drawImageToCanvas(image) {
        // fit image to canvas keeping aspect
        const cw = canvas.width, ch = canvas.height;
        const iw = image.width, ih = image.height;
        const scale = Math.min(cw / iw, ch / ih);
        const w = iw * scale, h = ih * scale;
        const x = (cw - w) / 2, y = (ch - h) / 2;
        ctx.clearRect(0,0,cw,ch);
        ctx.drawImage(image, x, y, w, h);
    }

    function pushHistory() {
        history.push(canvas.toDataURL());
        if (history.length > 20) history.shift();
    }

    undoBtn.addEventListener('click', () => {
        if (history.length === 0) return;
        const last = history.pop();
        const im = new Image();
        im.onload = () => { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.drawImage(im,0,0); };
        im.src = last;
    });

    resetBtn.addEventListener('click', () => {
        if (!originalImage) return;
        img = new Image();
        img.src = originalImage.src;
        img.onload = () => drawImageToCanvas(img);
        history = [];
    });

    exportBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = 'edited-image.png';
        document.body.appendChild(link);
        link.click();
        link.remove();
    });

    // open file
    fileInput.addEventListener('change', (e) => {
        const f = e.target.files[0];
        if (!f) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            originalImage = new Image();
            originalImage.onload = () => {
                img = originalImage;
                drawImageToCanvas(img);
                history = [];
            };
            originalImage.src = ev.target.result;
        };
        reader.readAsDataURL(f);
    });

    // simple rotate
    toolButtons.rotate.addEventListener('click', () => {
        if (!img.src) return;
        pushHistory();
        // rotate 90 degrees by drawing to temp canvas
        const temp = document.createElement('canvas');
        temp.width = img.height; temp.height = img.width;
        const tctx = temp.getContext('2d');
        tctx.translate(temp.width/2, temp.height/2);
        tctx.rotate(Math.PI/2);
        tctx.drawImage(img, -img.width/2, -img.height/2);
        img = new Image();
        img.onload = () => drawImageToCanvas(img);
        img.src = temp.toDataURL();
    });

    // flip horizontal
    toolButtons.flipH.addEventListener('click', () => {
        if (!img.src) return;
        pushHistory();
        const temp = document.createElement('canvas');
        temp.width = img.width; temp.height = img.height;
        const tctx = temp.getContext('2d');
        tctx.translate(temp.width, 0);
        tctx.scale(-1,1);
        tctx.drawImage(img, 0,0);
        img = new Image(); img.onload = () => drawImageToCanvas(img); img.src = temp.toDataURL();
    });

    // flip vertical
    toolButtons.flipV.addEventListener('click', () => {
        if (!img.src) return;
        pushHistory();
        const temp = document.createElement('canvas');
        temp.width = img.width; temp.height = img.height;
        const tctx = temp.getContext('2d');
        tctx.translate(0, temp.height);
        tctx.scale(1,-1);
        tctx.drawImage(img, 0,0);
        img = new Image(); img.onload = () => drawImageToCanvas(img); img.src = temp.toDataURL();
    });

    // resize tool
    toolButtons.resize.addEventListener('click', () => setActiveTool('resize'));
    document.getElementById('applyResize').addEventListener('click', () => {
        const w = parseInt(document.getElementById('resizeWidth').value,10);
        const h = parseInt(document.getElementById('resizeHeight').value,10);
        const keep = document.getElementById('resizeKeepAspect').checked;
        if (!img.src || !w || !h) return;
        pushHistory();
        const temp = document.createElement('canvas'); temp.width=w; temp.height=h;
        const tctx = temp.getContext('2d'); tctx.drawImage(img,0,0,w,h);
        img = new Image(); img.onload=()=>drawImageToCanvas(img); img.src = temp.toDataURL();
    });

    // select / crop tool (rectangle)
    let isDragging=false, startX=0, startY=0, selRect=null;
    const rectStyle = { stroke: 'rgba(0,212,255,0.9)', fill: 'rgba(0,212,255,0.08)' };
    toolButtons.select.addEventListener('click', () => setActiveTool('select'));

    canvas.addEventListener('mousedown', (e) => {
        if (currentTool === 'select') {
            const r = canvas.getBoundingClientRect(); startX = e.clientX - r.left; startY = e.clientY - r.top; isDragging=true; selRect={x:startX,y:startY,w:0,h:0};
        }
    });
    canvas.addEventListener('mousemove', (e) => {
        if (currentTool === 'select' && isDragging) {
            const r = canvas.getBoundingClientRect(); const mx = e.clientX - r.left; const my = e.clientY - r.top;
            selRect.w = mx - selRect.x; selRect.h = my - selRect.y;
            // redraw img + rect
            if (img.src) { drawImageToCanvas(img); drawSelection(); }
        }
    });
    canvas.addEventListener('mouseup', () => { if (currentTool==='select') isDragging=false; });

    function drawSelection(){
        if (!selRect) return;
        ctx.save(); ctx.beginPath(); ctx.fillStyle = rectStyle.fill; ctx.strokeStyle = rectStyle.stroke; ctx.lineWidth=2;
        ctx.fillRect(selRect.x, selRect.y, selRect.w, selRect.h); ctx.strokeRect(selRect.x, selRect.y, selRect.w, selRect.h);
        ctx.restore();
    }

    document.getElementById('applyCrop').addEventListener('click', () => {
        if (!selRect || !img.src) return;
        pushHistory();
        // create cropped image from canvas pixels
        const sx = Math.round(selRect.x), sy = Math.round(selRect.y), sw = Math.round(selRect.w), sh = Math.round(selRect.h);
        if (sw<=0 || sh<=0) return;
        const temp = document.createElement('canvas'); temp.width = Math.abs(sw); temp.height = Math.abs(sh);
        const tctx = temp.getContext('2d');
        // copy portion from current canvas
        tctx.drawImage(canvas, sx, sy, sw, sh, 0, 0, Math.abs(sw), Math.abs(sh));
        img = new Image(); img.onload = ()=> { drawImageToCanvas(img); selRect=null; }; img.src = temp.toDataURL();
    });

    // lasso: freehand drawing to create a mask
    let lassoPath = [];
    toolButtons.lasso.addEventListener('click', () => setActiveTool('lasso'));
    canvas.addEventListener('pointerdown', (e) => {
        if (currentTool!=='lasso') return; lassoPath=[]; canvas.setPointerCapture(e.pointerId); lassoPath.push(getPointerPos(e));
    });
    canvas.addEventListener('pointermove', (e) => { if (currentTool!=='lasso' || lassoPath.length===0) return; lassoPath.push(getPointerPos(e)); drawLassoPreview(); });
    canvas.addEventListener('pointerup', (e) => { if (currentTool!=='lasso') return; canvas.releasePointerCapture(e.pointerId); });

    function getPointerPos(e){ const r = canvas.getBoundingClientRect(); return {x: e.clientX - r.left, y: e.clientY - r.top}; }
    function drawLassoPreview(){ if (!img.src) return; drawImageToCanvas(img); if (lassoPath.length<2) return; ctx.save(); ctx.beginPath(); ctx.moveTo(lassoPath[0].x, lassoPath[0].y); for(let i=1;i<lassoPath.length;i++) ctx.lineTo(lassoPath[i].x, lassoPath[i].y); ctx.closePath(); ctx.fillStyle='rgba(0,212,255,0.08)'; ctx.fill(); ctx.strokeStyle='rgba(0,212,255,0.6)'; ctx.lineWidth=2; ctx.stroke(); ctx.restore(); }

    document.getElementById('applyLasso').addEventListener('click', () => {
        if (!img.src || lassoPath.length<3) return;
        pushHistory();
        // create mask canvas
        const mask = document.createElement('canvas'); mask.width=canvas.width; mask.height=canvas.height; const mctx = mask.getContext('2d');
        mctx.beginPath(); mctx.moveTo(lassoPath[0].x, lassoPath[0].y); for(let i=1;i<lassoPath.length;i++) mctx.lineTo(lassoPath[i].x, lassoPath[i].y); mctx.closePath(); mctx.fillStyle='#fff'; mctx.fill();
        // create resulting image by applying mask to current canvas drawing
        const temp = document.createElement('canvas'); temp.width = canvas.width; temp.height = canvas.height; const tctx = temp.getContext('2d');
        // draw original image to temp
        tctx.drawImage(canvas,0,0);
        // use mask to clear outside area
        tctx.globalCompositeOperation = 'destination-in'; tctx.drawImage(mask,0,0);
        // now scale down to bounding box of mask for nicer result
        // compute bbox
        const bbox = getMaskBBox(mask);
        if (!bbox) return;
        const out = document.createElement('canvas'); out.width = bbox.w; out.height = bbox.h; const octx = out.getContext('2d');
        octx.drawImage(temp, bbox.x, bbox.y, bbox.w, bbox.h, 0,0, bbox.w, bbox.h);
        img = new Image(); img.onload = ()=> drawImageToCanvas(img); img.src = out.toDataURL(); lassoPath=[];
    });

    function getMaskBBox(maskCanvas){ const w=maskCanvas.width,h=maskCanvas.height; const d = maskCanvas.getContext('2d').getImageData(0,0,w,h).data; let minX=w, minY=h, maxX=0, maxY=0, found=false; for(let y=0;y<h;y++){ for(let x=0;x<w;x++){ const i=(y*w + x)*4; if (d[i+3]>10){ found=true; if(x<minX)minX=x; if(y<minY)minY=y; if(x>maxX)maxX=x; if(y>maxY)maxY=y; } } } if(!found) return null; return {x:minX,y:minY,w:maxX-minX+1,h:maxY-minY+1}; }

    function drawLassoPreviewClear(){ lassoPath=[]; drawImageToCanvas(img); }

    // utility: set resize inputs when image present
    setInterval(()=>{
        if (!img || !img.src) return;
        const wIn = document.getElementById('resizeWidth'); const hIn = document.getElementById('resizeHeight'); if (wIn && hIn && img.width) { wIn.value = Math.round(img.width); hIn.value = Math.round(img.height); }
    },800);

    // tool button wiring
    toolButtons.select.addEventListener('click', ()=> setActiveTool('select'));
    toolButtons.lasso.addEventListener('click', ()=> setActiveTool('lasso'));

    // initial state
    setActiveTool(null);
});
