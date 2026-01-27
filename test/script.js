document.addEventListener('DOMContentLoaded', () => {
    // --- Global Elements ---
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const serverIpInput = document.getElementById('server-ip');
    const serverPortInput = document.getElementById('server-port');
    const clearBtn = document.getElementById('clear-btn');
    const saveBtn = document.getElementById('save-btn');

    // --- Mode Switching ---
    const imageModeBtn = document.getElementById('image-mode-btn');
    const videoModeBtn = document.getElementById('video-mode-btn');
    const imageControls = document.getElementById('image-controls');
    const videoControls = document.getElementById('video-controls');
    const videoPlayerContainer = document.getElementById('video-player-container');
    const videoResultsContainer = document.getElementById('video-results-container');

    // --- Image Mode Elements ---
    const imageLoader = document.getElementById('image-loader');
    const textBtn = document.getElementById('text-btn');
    const boxBtn = document.getElementById('box-btn');
    const textPromptContainer = document.getElementById('text-prompt-container');
    const textPromptInput = document.getElementById('text-prompt');
    const submitImageBtn = document.getElementById('submit-image-btn');
    
    // --- Auto Mode Elements (Image) ---
    const manualModeBtn = document.getElementById('manual-mode-btn');
    const autoModeBtn = document.getElementById('auto-mode-btn');
    const manualPromptContainer = document.getElementById('manual-prompt-container');
    const autoPromptContainer = document.getElementById('auto-prompt-container');

    // --- Video Mode Elements ---
    const videoLoader = document.getElementById('video-loader');
    const trackVideoBtn = document.getElementById('track-video-btn');
    const videoPlayer = document.getElementById('video-player');
    const frameSlider = document.getElementById('frame-slider');
    const frameLabel = document.getElementById('frame-label');
    
    // --- Auto Mode Elements (Video) ---
    const videoManualModeBtn = document.getElementById('video-manual-mode-btn');
    const videoAutoModeBtn = document.getElementById('video-auto-mode-btn');
    const videoManualPromptContainer = document.getElementById('video-manual-prompt-container');
    const videoAutoPromptContainer = document.getElementById('video-auto-prompt-container');

    // --- State ---
    let state = {
        globalMode: 'image', // 'image' or 'video'
        imageAnnotationMode: 'manual', // 'manual' or 'auto'
        videoAnnotationMode: 'manual', // 'manual' or 'auto'
        image: null,
        imgState: {
            modes: new Set(),
            boxes: [],
            isDrawing: false,
            startPoint: null,
            currentBox: null,
            resultMasks: [],
        },
        videoURL: null,
        videoBase64: null,
        videoBoxes: [],
        annotatedFrames: {}, // Will store { "0": Image, "1": Image, ... }
        videoTotalFrames: 0,
        videoFPS: 30, // Assuming a fixed FPS, might need adjustment
    };

    // ===================================================================
    // --- GENERAL FUNCTIONS ---
    // ===================================================================

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    // ===================================================================
    // --- IMAGE ANNOTATION MODE TOGGLE ---
    // ===================================================================

    manualModeBtn.addEventListener('click', () => {
        state.imageAnnotationMode = 'manual';
        manualModeBtn.classList.add('active');
        autoModeBtn.classList.remove('active');
        manualPromptContainer.style.display = 'block';
        autoPromptContainer.style.display = 'none';
        console.log('[Mode] Switched to manual image annotation mode');
    });

    autoModeBtn.addEventListener('click', () => {
        state.imageAnnotationMode = 'auto';
        autoModeBtn.classList.add('active');
        manualModeBtn.classList.remove('active');
        manualPromptContainer.style.display = 'none';
        autoPromptContainer.style.display = 'block';
        console.log('[Mode] Switched to auto image annotation mode');
    });

    // ===================================================================
    // --- VIDEO ANNOTATION MODE TOGGLE ---
    // ===================================================================

    videoManualModeBtn.addEventListener('click', () => {
        state.videoAnnotationMode = 'manual';
        videoManualModeBtn.classList.add('active');
        videoAutoModeBtn.classList.remove('active');
        videoManualPromptContainer.style.display = 'block';
        videoAutoPromptContainer.style.display = 'none';
        console.log('[Mode] Switched to manual video annotation mode');
    });

    videoAutoModeBtn.addEventListener('click', () => {
        state.videoAnnotationMode = 'auto';
        videoAutoModeBtn.classList.add('active');
        videoManualModeBtn.classList.remove('active');
        videoManualPromptContainer.style.display = 'none';
        videoAutoPromptContainer.style.display = 'block';
        console.log('[Mode] Switched to auto video annotation mode');
    });

    clearBtn.addEventListener('click', () => {
        // 只清除标注结果，保留已加载的图片/视频
        if (state.globalMode === 'image') {
            // 图像模式：清除标注结果但保留加载的图片
            state.imgState = {
                modes: new Set(),
                boxes: [],
                isDrawing: false,
                startPoint: null,
                currentBox: null,
                resultMasks: []  // 清除标注结果
            };
            textPromptInput.value = '';
            
            // 重新绘制原始图片（无标注）
            if (state.image) {
                redrawImageCanvas();
                console.log('[Clear] Image annotations cleared, original image displayed');
            }
        } else {
            // 视频模式：清除标注结果但保留加载的视频
            state.videoBoxes = [];
            state.annotatedFrames = {};
            state.videoTotalFrames = 0;
            videoResultsContainer.style.display = 'none';
            
            // 清除canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            console.log('[Clear] Video annotations cleared');
        }
    });

    saveBtn.addEventListener('click', () => {
        if (!state.image && Object.keys(state.annotatedFrames).length === 0) {
            alert('No image or video frame to save.');
            return;
        }
        const link = document.createElement('a');
        link.download = `annotated_${state.globalMode}_frame.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    });

    function switchGlobalMode(mode) {
        state.globalMode = mode;
        if (mode === 'image') {
            imageModeBtn.classList.add('active');
            videoModeBtn.classList.remove('active');
            imageControls.style.display = 'block';
            videoControls.style.display = 'none';
            videoResultsContainer.style.display = 'none';
            videoPlayerContainer.style.display = 'none';
        } else {
            videoModeBtn.classList.add('active');
            imageModeBtn.classList.remove('active');
            videoControls.style.display = 'block';
            imageControls.style.display = 'none';
        }
        clearBtn.click();
    }

    imageModeBtn.addEventListener('click', () => switchGlobalMode('image'));
    videoModeBtn.addEventListener('click', () => switchGlobalMode('video'));

    // ===================================================================
    // --- IMAGE MODE LOGIC ---
    // ===================================================================

    imageLoader.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (event) => {
            state.image = new Image();
            state.image.onload = () => {
                const ratio = Math.min(800 / state.image.width, 600 / state.image.height);
                canvas.width = state.image.width * ratio;
                canvas.height = state.image.height * ratio;
                drawImageScaled(state.image, canvas);
            };
            state.image.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });

    function setImageMode(newMode) {
        if (state.imgState.modes.has(newMode)) state.imgState.modes.delete(newMode);
        else state.imgState.modes.add(newMode);
        updateImageButtons();
    }

    function updateImageButtons() {
        [textBtn, boxBtn].forEach(btn => btn.classList.remove('active'));
        textPromptContainer.style.display = 'none';
        state.imgState.modes.forEach(mode => {
            if (mode === 'text') {
                textBtn.classList.add('active');
                textPromptContainer.style.display = 'block';
            } else if (mode === 'box') {
                boxBtn.classList.add('active');
            }
        });
    }

    textBtn.addEventListener('click', () => setImageMode('text'));
    boxBtn.addEventListener('click', () => setImageMode('box'));

    submitImageBtn.addEventListener('click', async () => {
        if (!state.image) return alert('Please load an image.');
        
        let apiUrl;
        let payload = { image_base64: getBase64FromImage(state.image) };
        
        if (state.imageAnnotationMode === 'auto') {
            // 自动模式：无需任何提示
            apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict_auto`;
        } else {
            // 手动模式：需要提示
            let hasPrompts = false;
            if (textPromptInput.value) {
                payload.texts = [{ text: textPromptInput.value }];
                hasPrompts = true;
            }
            if (state.imgState.boxes.length > 0) {
                payload.boxes = state.imgState.boxes;
                hasPrompts = true;
            }
            if (!hasPrompts) return alert('Please provide a text or box prompt.');
            apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict`;
        }
        
        await submitRequest(apiUrl, payload, 'image');
    });

    // ===================================================================
    // --- VIDEO MODE LOGIC ---
    // ===================================================================

    videoLoader.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            state.videoBase64 = event.target.result.split(',')[1];
        };
        reader.readAsDataURL(file);

        state.videoURL = URL.createObjectURL(file);
        videoPlayer.src = state.videoURL;
        videoPlayer.onloadedmetadata = () => {
            const ratio = Math.min(800 / videoPlayer.videoWidth, 600 / videoPlayer.videoHeight);
            canvas.width = videoPlayer.videoWidth * ratio;
            canvas.height = videoPlayer.videoHeight * ratio;
            videoPlayer.onseeked = () => {
                drawImageScaled(videoPlayer, canvas);
            };
            videoPlayer.currentTime = 0;
        };
    });

    trackVideoBtn.addEventListener('click', async () => {
        if (!state.videoBase64) return alert('Please load a video.');
        
        let apiUrl;
        let payload = { video_base64: state.videoBase64 };
        
        if (state.videoAnnotationMode === 'auto') {
            // 自动模式：无需任何提示
            apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict_video_auto`;
        } else {
            // 手动模式：需要框选提示
            if (state.videoBoxes.length === 0) return alert('Please add at least one box prompt on the first frame.');
            payload.boxes = state.videoBoxes;
            apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict_video`;
        }
        
        await submitRequest(apiUrl, payload, 'video');
    });

    frameSlider.addEventListener('input', (e) => {
        const frame = parseInt(e.target.value);
        frameLabel.textContent = frame;
        console.log(`[Slider] Moving to frame ${frame}, checking state.annotatedFrames...`);
        console.log(`[Slider] Available frames:`, Object.keys(state.annotatedFrames));
        
        const frameKey = String(frame);
        if (frameKey in state.annotatedFrames) {
            console.log(`[Slider] Frame ${frameKey} found, drawing...`);
            drawVideoFrame(frame);
        } else {
            console.warn(`[Slider] Frame ${frameKey} not loaded yet. Available: ${Object.keys(state.annotatedFrames).join(', ')}`);
        }
    });

    // ===================================================================
    // --- CANVAS & DRAWING LOGIC ---
    // ===================================================================

    canvas.addEventListener('mousedown', (e) => {
        const canDrawBox = state.globalMode === 'image' && state.imgState.modes.has('box') && state.imageAnnotationMode === 'manual';
        const canDrawVideoBox = state.globalMode === 'video' && videoPlayer.src && state.videoAnnotationMode === 'manual';
        if (canDrawBox || canDrawVideoBox) {
            state.imgState.isDrawing = true;
            state.imgState.startPoint = getCanvasCoords(e);
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!state.imgState.isDrawing) return;
        const currentPoint = getCanvasCoords(e);
        const p1 = state.imgState.startPoint;
        state.imgState.currentBox = { x: Math.min(p1.x, currentPoint.x), y: Math.min(p1.y, currentPoint.y), w: Math.abs(p1.x - currentPoint.x), h: Math.abs(p1.y - currentPoint.y) };
        if (state.globalMode === 'image') {
            redrawImageCanvas();
        } else {
            // Draw original video frame with the temp box
            drawImageScaled(videoPlayer, canvas);
            drawBoxes(state.videoBoxes, canvas.width / videoPlayer.videoWidth, 'blue');
            if (state.imgState.currentBox) {
                ctx.strokeStyle = 'red';
                ctx.strokeRect(state.imgState.currentBox.x, state.imgState.currentBox.y, state.imgState.currentBox.w, state.imgState.currentBox.h);
            }
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (!state.imgState.isDrawing) return;
        state.imgState.isDrawing = false;
        const p1 = state.imgState.startPoint;
        const p2 = getCanvasCoords(e);
        const scale = (state.globalMode === 'image') ? (canvas.width / state.image.width) : (canvas.width / videoPlayer.videoWidth);
        const newBox = { box: [Math.min(p1.x, p2.x) / scale, Math.min(p1.y, p2.y) / scale, Math.max(p1.x, p2.x) / scale, Math.max(p1.y, p2.y) / scale] };
        
        if (state.globalMode === 'image') {
            state.imgState.boxes.push(newBox);
            state.imgState.currentBox = null;
            redrawImageCanvas();
        } else if (state.globalMode === 'video') {
            state.videoBoxes.push(newBox);
            state.imgState.currentBox = null;
            drawImageScaled(videoPlayer, canvas);
            drawBoxes(state.videoBoxes, scale, 'blue');
        }
    });

    function drawImageScaled(source, targetCanvas) {
        const targetCtx = targetCanvas.getContext('2d');
        targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        targetCtx.drawImage(source, 0, 0, targetCanvas.width, targetCanvas.height);
    }

    function getBase64FromImage(img) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = img.naturalWidth;
        tempCanvas.height = img.naturalHeight;
        tempCanvas.getContext('2d').drawImage(img, 0, 0);
        return tempCanvas.toDataURL('image/jpeg').split(',')[1];
    }

    function drawBoxes(boxList, scale, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        boxList.forEach(b => {
            ctx.strokeRect(b.box[0] * scale, b.box[1] * scale, (b.box[2] - b.box[0]) * scale, (b.box[3] - b.box[1]) * scale);
        });
    }

    function redrawImageCanvas() {
        if (!state.image) return;
        drawImageScaled(state.image, canvas);
        drawBoxes(state.imgState.boxes, canvas.width / state.image.width, 'blue');
        if (state.imgState.currentBox) {
            ctx.strokeStyle = 'red';
            ctx.strokeRect(state.imgState.currentBox.x, state.imgState.currentBox.y, state.imgState.currentBox.w, state.imgState.currentBox.h);
        }
        state.imgState.resultMasks.forEach(maskObj => {
            if (maskObj.img) ctx.drawImage(maskObj.img, 0, 0, canvas.width, canvas.height);
        });
    }

    function drawVideoFrame(frameIndex) {
        const frameKey = String(frameIndex);
        console.log(`[drawVideoFrame] Drawing frame ${frameIndex} (key: '${frameKey}')`);
        console.log(`[drawVideoFrame] Frame exists:`, frameKey in state.annotatedFrames);
        
        const annotatedImage = state.annotatedFrames[frameKey];
        if (annotatedImage) {
            console.log(`[drawVideoFrame] Image found. Complete: ${annotatedImage.complete}, Width: ${annotatedImage.width}, Height: ${annotatedImage.height}`);
            if (annotatedImage.complete) {
                drawImageScaled(annotatedImage, canvas);
                console.log(`[drawVideoFrame] ✓ Successfully drew frame ${frameIndex} on canvas`);
            } else {
                console.warn(`[drawVideoFrame] Image not complete yet for frame ${frameIndex}`);
                // Wait a bit and retry
                setTimeout(() => {
                    if (annotatedImage.complete) {
                        drawImageScaled(annotatedImage, canvas);
                        console.log(`[drawVideoFrame] ✓ Drew frame ${frameIndex} after retry`);
                    }
                }, 100);
            }
        } else {
            console.warn(`[drawVideoFrame] ✗ Frame ${frameIndex} not found in state.annotatedFrames`);
            console.warn(`[drawVideoFrame] Available frames:`, Object.keys(state.annotatedFrames));
        }
    }

    // ===================================================================
    // --- SERVER COMMUNICATION ---
    // ===================================================================

    async function submitRequest(apiUrl, payload, type) {
        const btn = type === 'image' ? submitImageBtn : trackVideoBtn;
        console.log('Sending payload:', payload);
        try {
            btn.disabled = true;
            btn.textContent = 'Processing...';
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log('Received data:', data);
            if (type === 'image') {
                await processImageResponse(data);
            } else {
                await processVideoResponse(data);
            }
        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = type === 'image' ? 'Annotate Image' : 'Track Objects in Video';
        }
    }

    async function processImageResponse(data) {
        state.imgState.resultMasks = data.masks || [];
        await loadMaskImages(state.imgState.resultMasks);
        redrawImageCanvas();
    }

    async function processVideoResponse(data) {
        const frameData = data.frames || {};
        state.videoTotalFrames = Object.keys(frameData).length;
        
        console.log(`[Video Response] Total frames: ${state.videoTotalFrames}`);
        console.log(`[Video Response] Frame keys:`, Object.keys(frameData));

        videoResultsContainer.style.display = 'block';
        frameSlider.max = state.videoTotalFrames > 0 ? state.videoTotalFrames - 1 : 0;
        frameSlider.value = 0;
        frameLabel.textContent = '0';
        
        // Update total frames display if exists
        const frameTotal = document.getElementById('frame-total');
        if (frameTotal) {
            frameTotal.textContent = state.videoTotalFrames > 0 ? state.videoTotalFrames - 1 : 0;
        }
        
        console.log(`[Video Response] Loading ${state.videoTotalFrames} frames...`);
        await loadAnnotatedFrames(frameData);
        console.log(`[Video Response] Frames loaded. Drawing frame 0...`);

        // Draw the first annotated frame
        drawVideoFrame(0);
    }
    
    function loadMaskImages(maskList) {
        const promises = maskList.map(maskObj => {
            return new Promise((resolve) => {
                if (maskObj.img) return resolve();
                const maskImage = new Image();
                maskImage.src = maskObj.mask_base64;
                maskImage.onload = () => {
                    maskObj.img = maskImage;
                    resolve();
                };
                maskImage.onerror = () => resolve();
            });
        });
        return Promise.all(promises);
    }
    
    function loadAnnotatedFrames(frameData) {
        console.log(`[loadAnnotatedFrames] Loading ${Object.keys(frameData).length} frames`);
        const promises = Object.entries(frameData).map(([frameIndex, base64String]) => {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    state.annotatedFrames[frameIndex] = img;
                    console.log(`[loadAnnotatedFrames] ✓ Loaded frame ${frameIndex} (${img.width}x${img.height})`);
                    resolve();
                };
                img.onerror = (err) => {
                    console.error(`[loadAnnotatedFrames] ✗ Failed to load frame ${frameIndex}:`, err);
                    resolve(); // Continue even if one frame fails
                };
                img.src = base64String;
            });
        });
        return Promise.all(promises).then(() => {
            console.log(`[loadAnnotatedFrames] All frames loaded. Total: ${Object.keys(state.annotatedFrames).length}`);
        });
    }
});