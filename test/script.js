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

    // --- Video Mode Elements ---
    const videoLoader = document.getElementById('video-loader');
    const trackVideoBtn = document.getElementById('track-video-btn');
    const videoPlayer = document.getElementById('video-player');
    const frameSlider = document.getElementById('frame-slider');
    const frameLabel = document.getElementById('frame-label');
    
    // --- Video Text Tracking Elements ---
    const useTextPromptBtn = document.getElementById('use-text-prompt-btn');
    const useBoxPromptBtn = document.getElementById('use-box-prompt-btn');
    const videoTextPromptContainer = document.getElementById('video-text-prompt-container');
    const videoBoxPromptContainer = document.getElementById('video-box-prompt-container');
    const videoTextPromptInput = document.getElementById('video-text-prompt');

    // --- State ---
    let state = {
        globalMode: 'image', // 'image' or 'video'
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
        videoTextPrompt: '',
        annotatedFrames: {}, // Will store { "0": Image, "1": Image, ... }
        videoTotalFrames: 0,
        videoFPS: 30, // Assuming a fixed FPS, might need adjustment
    };

    // ===================================================================
    // --- GENERAL FUNCTIONS ---
    // ===================================================================
    
        // ===================================================================
        // --- MODE SWITCHING (Image vs Video) ---
        // ===================================================================
    
        imageModeBtn.addEventListener('click', () => {
            state.globalMode = 'image';
            imageModeBtn.classList.add('active');
            videoModeBtn.classList.remove('active');
            imageControls.style.display = 'block';
            videoControls.style.display = 'none';
            videoPlayerContainer.style.display = 'none';
            videoResultsContainer.style.display = 'none';
            canvas.style.display = 'block'; // Ensure canvas is visible
        
            // If an image is loaded, redraw it
            if (state.image) {
                redrawImageCanvas();
            } else {
                // Clear canvas if no image
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            console.log('[Mode] Switched to image mode');
        });
    
        videoModeBtn.addEventListener('click', () => {
            state.globalMode = 'video';
            videoModeBtn.classList.add('active');
            imageModeBtn.classList.remove('active');
            imageControls.style.display = 'none';
            videoControls.style.display = 'block';
            videoPlayerContainer.style.display = 'none'; // Always hide the video player element
            canvas.style.display = 'block'; // Ensure canvas is visible
    
            // If a video is loaded, draw its first frame on the canvas
            if (state.videoURL) {
                canvas.width = videoPlayer.videoWidth;
                canvas.height = videoPlayer.videoHeight;
                drawImageScaled(videoPlayer, canvas);
                redrawVideoCanvas(); // Also draw any existing boxes
            } else {
                // Clear canvas if no video
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
            console.log('[Mode] Switched to video mode');
        });
    
        // ===================================================================
        // --- VIDEO TRACKING MODE TOGGLE (Text vs Box) ---
        // ===================================================================
    
        useTextPromptBtn.addEventListener('click', () => {
            useTextPromptBtn.classList.add('active');
            useBoxPromptBtn.classList.remove('active');
            videoTextPromptContainer.style.display = 'block';
            videoBoxPromptContainer.style.display = 'none';
            console.log('[Video Mode] Switched to text prompt tracking');
        });
    
        useBoxPromptBtn.addEventListener('click', () => {
            useBoxPromptBtn.classList.add('active');
            useTextPromptBtn.classList.remove('active');
            videoTextPromptContainer.style.display = 'none';
            videoBoxPromptContainer.style.display = 'block';
            console.log('[Video Mode] Switched to box prompt tracking');
        });
    
        // ===================================================================
        // --- IMAGE MODE LOGIC ---
        // ===================================================================
    
        imageLoader.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
    
            state.image = new Image();
            state.image.onload = () => {
                canvas.width = state.image.width;
                canvas.height = state.image.height;
                state.imgState.boxes = [];
                state.imgState.resultMasks = [];
                redrawImageCanvas();
            };
            const reader = new FileReader();
            reader.onload = (event) => {
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
            
            let hasPrompts = false;
            const payload = { image_base64: getBase64FromImage(state.image) };
            
            if (textPromptInput.value) {
                payload.texts = [{ text: textPromptInput.value }];
                hasPrompts = true;
            }
            if (state.imgState.boxes.length > 0) {
                payload.boxes = state.imgState.boxes;
                hasPrompts = true;
            }
            if (!hasPrompts) return alert('Please provide a text or box prompt.');
            
            const apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict`;
            await submitRequest(apiUrl, payload, 'image');
        });
    
        // ===================================================================
        // --- VIDEO MODE LOGIC ---
        // ===================================================================
    
        videoLoader.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
    
            // Reset video-related state
            state.videoBoxes = [];
            state.annotatedFrames = {};
    
            const reader = new FileReader();
            reader.onload = (event) => {
                state.videoBase64 = event.target.result.split(',')[1];
            };
            reader.readAsDataURL(file);
    
            state.videoURL = URL.createObjectURL(file);
            videoPlayer.src = state.videoURL;
    
            videoPlayer.onloadedmetadata = () => {
                // Set canvas drawing buffer size to match video's original dimensions
                canvas.width = videoPlayer.videoWidth;
                canvas.height = videoPlayer.videoHeight;
                
                // Hide the actual video player and show the canvas
                videoPlayerContainer.style.display = 'none';
                canvas.style.display = 'block';
    
                videoPlayer.onseeked = () => {
                    // Draw the current video frame to the canvas
                    redrawVideoCanvas();
                };
                // Seek to beginning to show the first frame
                videoPlayer.currentTime = 0.01; 
            };
        });
    
        trackVideoBtn.addEventListener('click', async () => {
            if (!state.videoBase64) return alert('Please load a video.');
            
            // Check if using text prompt or box prompt
            if (videoTextPromptInput.value && videoTextPromptInput.value.trim() !== '') {
                // Text-based tracking
                const payload = {
                    video_base64: state.videoBase64,
                    text_prompt: videoTextPromptInput.value.trim()
                };
                const apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict_video_text`;
                await submitRequest(apiUrl, payload, 'video');
            } else if (state.videoBoxes.length > 0) {
                // Box-based tracking
                const payload = {
                    video_base64: state.videoBase64,
                    boxes: state.videoBoxes
                };
                const apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict_video`;
                await submitRequest(apiUrl, payload, 'video');
            } else {
                alert('Please either:\n1. Enter a text prompt to track objects by description\n2. Or draw boxes on the first frame to track specific regions');
            }
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
        // --- CANVAS & DRAWING LOGIC (REFACTORED) ---
        // ===================================================================
    
        function getCanvasCoords(e) {
            // Using offsetX/Y is the most robust way to get coordinates relative to the padding edge.
            return { x: e.offsetX, y: e.offsetY };
        }
    function getCurrentMedia() {
        if (state.globalMode === 'image' && state.image) {
            return {
                media: state.image,
                width: state.image.width,
                height: state.image.height,
            };
        }
        if (state.globalMode === 'video' && state.videoURL) {
            return {
                media: videoPlayer,
                width: videoPlayer.videoWidth,
                height: videoPlayer.videoHeight,
            };
        }
        return null;
    }

    canvas.addEventListener('mousedown', (e) => {
        const mediaInfo = getCurrentMedia();
        if (!mediaInfo) return;

        const canDrawBox = (state.globalMode === 'image' && state.imgState.modes.has('box')) ||
                           (state.globalMode === 'video' && useBoxPromptBtn.classList.contains('active'));

        if (canDrawBox) {
            state.imgState.isDrawing = true;
            // Store start point in canvas-space coordinates
            state.imgState.startPoint = getCanvasCoords(e);
            state.imgState.currentBox = null; // Reset current box
        }
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!state.imgState.isDrawing) return;

        const startPoint = state.imgState.startPoint;
        const currentPoint = getCanvasCoords(e);

        // Define the box in canvas-space coordinates
        state.imgState.currentBox = {
            x: Math.min(startPoint.x, currentPoint.x),
            y: Math.min(startPoint.y, currentPoint.y),
            w: Math.abs(startPoint.x - currentPoint.x),
            h: Math.abs(startPoint.y - currentPoint.y),
        };

        // Redraw based on mode
        if (state.globalMode === 'image') {
            redrawImageCanvas();
        } else {
            redrawVideoCanvas();
        }
    });

    canvas.addEventListener('mouseup', (e) => {
        if (!state.imgState.isDrawing) return;
        state.imgState.isDrawing = false;

        const mediaInfo = getCurrentMedia();
        if (!mediaInfo) return;

        // Final box in canvas-space coordinates
        const endPoint = getCanvasCoords(e);
        const startPoint = state.imgState.startPoint;
        const canvasBox = {
            x1: Math.min(startPoint.x, endPoint.x),
            y1: Math.min(startPoint.y, endPoint.y),
            x2: Math.max(startPoint.x, endPoint.x),
            y2: Math.max(startPoint.y, endPoint.y),
        };
        
        // The canvas buffer is sized to the media, but the canvas element may be scaled by CSS.
        // We need to convert the box from the canvas's client-space (CSS pixels) to its buffer-space.
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        // Convert to original media coordinates and save
        const newBox = {
            box: [
                canvasBox.x1 * scaleX,
                canvasBox.y1 * scaleY,
                canvasBox.x2 * scaleX,
                canvasBox.y2 * scaleY,
            ]
        };

        if (state.globalMode === 'image') {
            state.imgState.boxes.push(newBox);
            redrawImageCanvas();
        } else if (state.globalMode === 'video') {
            state.videoBoxes.push(newBox);
            redrawVideoCanvas();
        }
        state.imgState.currentBox = null; // Clear temp box
    });

    function drawImageScaled(source, targetCanvas) {
        const targetCtx = targetCanvas.getContext('2d');
        targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        
        // The canvas buffer has the same size as the source media, so we draw 1:1
        targetCtx.drawImage(source, 0, 0, targetCanvas.width, targetCanvas.height);
    }

    function getBase64FromImage(img) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = img.naturalWidth || img.width;
        tempCanvas.height = img.naturalHeight || img.height;
        tempCanvas.getContext('2d').drawImage(img, 0, 0);
        return tempCanvas.toDataURL('image/jpeg').split(',')[1];
    }

    function drawBoxes(boxList, color) {
        const mediaInfo = getCurrentMedia();
        if (!mediaInfo) return;

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            boxList.forEach(b => {
                ctx.strokeRect(
                    b.box[0],
                    b.box[1],
                    b.box[2] - b.box[0],
                    b.box[3] - b.box[1]
                );
            });    }

    function redrawImageCanvas() {
        if (!state.image) return;
        drawImageScaled(state.image, canvas);
        drawBoxes(state.imgState.boxes, 'blue');

        // Draw the temporary box, scaling from CSS-space to canvas-space
        if (state.imgState.currentBox) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                state.imgState.currentBox.x * scaleX,
                state.imgState.currentBox.y * scaleY,
                state.imgState.currentBox.w * scaleX,
                state.imgState.currentBox.h * scaleY
            );
        }
    }
    
    function redrawVideoCanvas() {
        if (!videoPlayer.src) return;
        drawImageScaled(videoPlayer, canvas);
        drawBoxes(state.videoBoxes, 'blue');
        
        // Draw the temporary box, scaling from CSS-space to canvas-space
        if (state.imgState.currentBox) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(
                state.imgState.currentBox.x * scaleX,
                state.imgState.currentBox.y * scaleY,
                state.imgState.currentBox.w * scaleX,
                state.imgState.currentBox.h * scaleY
            );
        }
    }

    function drawVideoFrame(frameIndex) {
        const frameKey = String(frameIndex);
        console.log(`[drawVideoFrame] Drawing frame ${frameIndex} (key: '${frameKey}')`);
        console.log(`[drawVideoFrame] Frame exists:`, frameKey in state.annotatedFrames);
        
        const annotatedImage = state.annotatedFrames[frameKey];
        if (annotatedImage) {
            console.log(`[drawVideoFrame] Image found. Complete: ${annotatedImage.complete}, Width: ${annotatedImage.width}, Height: ${annotatedImage.height}`);
            if (annotatedImage.complete) {
                // Ensure canvas size matches the frame before drawing
                canvas.width = annotatedImage.width;
                canvas.height = annotatedImage.height;
                drawImageScaled(annotatedImage, canvas);
                console.log(`[drawVideoFrame] ✓ Successfully drew frame ${frameIndex} on canvas`);
            } else {
                console.warn(`[drawVideoFrame] Image not complete yet for frame ${frameIndex}`);
                annotatedImage.onload = () => { // Add onload handler for race conditions
                    canvas.width = annotatedImage.width;
                    canvas.height = annotatedImage.height;
                    drawImageScaled(annotatedImage, canvas);
                    console.log(`[drawVideoFrame] ✓ Drew frame ${frameIndex} after onload`);
                };
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
        console.log('[processImageResponse] Received data:', data);
        state.imgState.resultMasks = data.masks || [];
        console.log('[processImageResponse] Masks count:', state.imgState.resultMasks.length);
        
        if (state.imgState.resultMasks.length === 0) {
            alert('No masks detected. Please try a different prompt or image.');
            return;
        }
        
        await loadMaskImages(state.imgState.resultMasks);
        console.log('[processImageResponse] All masks loaded, redrawing canvas');
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
        const promises = maskList.map((maskObj, index) => {
            return new Promise((resolve) => {
                // Skip if already loaded
                if (maskObj.img) {
                    console.log(`[loadMaskImages] Mask ${index} already loaded`);
                    return resolve();
                }
                
                // Create Image object from base64
                const maskImage = new Image();
                maskImage.onload = () => {
                    maskObj.img = maskImage;
                    console.log(`[loadMaskImages] ✓ Loaded mask ${index}: ${maskObj.label}`);
                    resolve();
                };
                maskImage.onerror = (err) => {
                    console.error(`[loadMaskImages] ✗ Failed to load mask ${index}:`, err);
                    resolve(); // Continue even if one mask fails
                };
                
                // Set the base64 data URL
                maskImage.src = maskObj.mask_base64;
                console.log(`[loadMaskImages] Loading mask ${index}: ${maskObj.label}`);
            });
        });
        return Promise.all(promises).then(() => {
            console.log(`[loadMaskImages] All ${maskList.length} masks loaded successfully`);
        });
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

    // ===================================================================
    // --- UI CONTROLS ---
    // ===================================================================

    clearBtn.addEventListener('click', () => {
        if (state.globalMode === 'image') {
            state.imgState.boxes = [];
            state.imgState.resultMasks = [];
            redrawImageCanvas();
        } else {
            state.videoBoxes = [];
            videoTextPromptInput.value = '';
            state.annotatedFrames = {};
            frameSlider.value = 0;
            frameLabel.textContent = '0';
            if (state.videoURL) {
                drawImageScaled(videoPlayer, canvas);
            }
        }
        console.log('[UI] Cleared');
    });

    saveBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = `sam3-${Date.now()}.png`;
        link.click();
        console.log('[UI] Image saved');
    });

    // Initialize with image mode
    imageModeBtn.click();
});
