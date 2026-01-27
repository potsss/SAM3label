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

    // --- State ---
    let state = {
        globalMode: 'image', // 'image' or 'video'
        // Image state
        image: null,
        imgState: {
            modes: new Set(),
            boxes: [],
            isDrawing: false,
            startPoint: null,
            currentBox: null,
            resultMasks: [],
        },
        // Video state
        video: null,
        videoURL: null,
        videoBase64: null,
        videoBoxes: [],
        videoFrameData: {},
        videoTotalFrames: 0,
        videoFPS: 30,
    };

    // ===================================================================
    // --- GENERAL FUNCTIONS ---
    // ===================================================================

    function getCanvasCoords(e) {
        const rect = canvas.getBoundingClientRect();
        return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }

    clearBtn.addEventListener('click', () => {
        // Reset all state
        state.image = null;
        state.imgState = { modes: new Set(), boxes: [], isDrawing: false, startPoint: null, currentBox: null, resultMasks: [] };
        state.video = null;
        state.videoURL = null;
        state.videoBase64 = null;
        state.videoBoxes = [];
        state.videoFrameData = {};
        // Clear UI
        textPromptInput.value = '';
        imageLoader.value = '';
        videoLoader.value = '';
        videoResultsContainer.style.display = 'none';
        videoPlayer.src = '';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });

    saveBtn.addEventListener('click', () => {
        if (!state.image && !state.video) {
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
        clearBtn.click(); // Clear state when switching modes
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
        
        const payload = { image_base64: getBase64FromImage(state.image) };
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
        
        const apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict`;
        await submitRequest(apiUrl, payload, 'image');
    });


    // ===================================================================
    // --- VIDEO MODE LOGIC ---
    // ===================================================================

    videoLoader.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        // Store file for base64 conversion on submit
        const reader = new FileReader();
        reader.onload = (event) => {
            state.videoBase64 = event.target.result.split(',')[1];
        };
        reader.readAsDataURL(file);

        // Create object URL for playback and first-frame capture
        state.videoURL = URL.createObjectURL(file);
        videoPlayer.src = state.videoURL;
        videoPlayer.onloadedmetadata = () => {
            const ratio = Math.min(800 / videoPlayer.videoWidth, 600 / videoPlayer.videoHeight);
            canvas.width = videoPlayer.videoWidth * ratio;
            canvas.height = videoPlayer.videoHeight * ratio;
            videoPlayer.onseeked = () => {
                drawImageScaled(videoPlayer, canvas);
            };
            videoPlayer.currentTime = 0; // Seek to first frame
        };
    });

        trackVideoBtn.addEventListener('click', async () => {

            if (!state.videoBase64) return alert('Please load a video.');

            if (state.videoBoxes.length === 0) return alert('Please add at least one box prompt on the first frame.');

    

            const payload = {

                video_base64: state.videoBase64,

                boxes: state.videoBoxes,

            };

            

            const apiUrl = `http://${serverIpInput.value}:${serverPortInput.value}/predict_video`;

            await submitRequest(apiUrl, payload, 'video');

        });

        

        frameSlider.addEventListener('input', (e) => {

            const frame = parseInt(e.target.value);

            frameLabel.textContent = frame;

            const time = frame / state.videoFPS;

            if (videoPlayer.src && Math.abs(videoPlayer.currentTime - time) > 0.1) {

                videoPlayer.currentTime = time;

            }

        });

    

        videoPlayer.addEventListener('timeupdate', () => {

            drawVideoFrame();

        });

    

    

        // ===================================================================

        // --- CANVAS & DRAWING LOGIC ---

        // ===================================================================

    

        canvas.addEventListener('mousedown', (e) => {

            const canDrawBox = state.globalMode === 'image' && state.imgState.modes.has('box');

            const canDrawVideoBox = state.globalMode === 'video' && videoPlayer.src;

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

                drawVideoFrame(); // Redraw video frame with the temp box

            }

        });

    

        canvas.addEventListener('mouseup', (e) => {

            if (!state.imgState.isDrawing) return;

            state.imgState.isDrawing = false;

    

            const p1 = state.imgState.startPoint;

            const p2 = getCanvasCoords(e);

    

            if (state.globalMode === 'image') {

                const scale = canvas.width / state.image.width;

                state.imgState.boxes.push({ box: [Math.min(p1.x, p2.x)/scale, Math.min(p1.y, p2.y)/scale, Math.max(p1.x, p2.x)/scale, Math.max(p1.y, p2.y)/scale] });

                state.imgState.currentBox = null;

                redrawImageCanvas();

            } else if (state.globalMode === 'video') {

                const scale = canvas.width / videoPlayer.videoWidth;

                state.videoBoxes.push({ box: [Math.min(p1.x, p2.x)/scale, Math.min(p1.y, p2.y)/scale, Math.max(p1.x, p2.x)/scale, Math.max(p1.y, p2.y)/scale] });

                state.imgState.currentBox = null;

                drawVideoFrame();

            }

        });

    

        // This listener is now replaced by the mouse up/down/move logic for boxes

        // canvas.addEventListener('click', (e) => { ... });

    

        function drawImageScaled(source, targetCanvas) {

            const targetCtx = targetCanvas.getContext('2d');

            targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);

            targetCtx.drawImage(source, 0, 0, targetCanvas.width, targetCanvas.height);

        }

        

        function getBase64FromImage(img) {

            const tempCanvas = document.createElement('canvas');

            tempCanvas.width = img.naturalWidth;

            tempCanvas.height = img.naturalHeight;

            const tempCtx = tempCanvas.getContext('2d');

            tempCtx.drawImage(img, 0, 0);

            return tempCanvas.toDataURL('image/jpeg').split(',')[1];

        }

        

        function redrawImageCanvas() {

            if (!state.image) return;

            drawImageScaled(state.image, canvas);

    

            // Draw prompt boxes

            const scale = canvas.width / state.image.width;

            ctx.strokeStyle = 'blue';

            ctx.lineWidth = 2;

            state.imgState.boxes.forEach(b => {

                ctx.strokeRect(b.box[0] * scale, b.box[1] * scale, (b.box[2] - b.box[0]) * scale, (b.box[3] - b.box[1]) * scale);

            });

            if (state.imgState.currentBox) {

                ctx.strokeStyle = 'red';

                ctx.strokeRect(state.imgState.currentBox.x, state.imgState.currentBox.y, state.imgState.currentBox.w, state.imgState.currentBox.h);

            }

    

            // Draw result masks

            state.imgState.resultMasks.forEach(maskObj => {

                if (maskObj.img) ctx.drawImage(maskObj.img, 0, 0, canvas.width, canvas.height);

            });

        }

    

        function drawVideoFrame() {

            if (!videoPlayer.src) return;

            drawImageScaled(videoPlayer, canvas);

    

            // Draw prompt boxes on video frame

            const scale = canvas.width / videoPlayer.videoWidth;

            ctx.strokeStyle = 'blue';

            ctx.lineWidth = 2;

            state.videoBoxes.forEach(b => {

                ctx.strokeRect(b.box[0] * scale, b.box[1] * scale, (b.box[2] - b.box[0]) * scale, (b.box[3] - b.box[1]) * scale);

            });

            if (state.imgState.currentBox) {

                ctx.strokeStyle = 'red';

                ctx.strokeRect(state.imgState.currentBox.x, state.imgState.currentBox.y, state.imgState.currentBox.w, state.imgState.currentBox.h);

            }

    

            // Draw masks for current video frame

            const frame = Math.floor(videoPlayer.currentTime * state.videoFPS);

            const frameMasks = state.videoFrameData[frame] || [];

            frameMasks.forEach(maskObj => {

                if (maskObj.img) ctx.drawImage(maskObj.img, 0, 0, canvas.width, canvas.height);

            });

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
        state.videoFrameData = data.frames || {};
        videoResultsContainer.style.display = 'block';

        // Get video metadata
        const duration = videoPlayer.duration;
        const totalFrames = Math.floor(duration * state.videoFPS); // Approximate
        state.videoTotalFrames = totalFrames;
        
        frameSlider.max = totalFrames > 0 ? totalFrames - 1 : 0;
        frameSlider.value = 0;
        frameLabel.textContent = '0';

        // Pre-load all mask images for smooth scrubbing
        const allMasks = Object.values(state.videoFrameData).flat();
        await loadMaskImages(allMasks);

        // Start playback and drawing
        videoPlayer.currentTime = 0;
        drawVideoFrame();
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
                maskImage.onerror = () => resolve(); // Continue even if one mask fails
            });
        });
        return Promise.all(promises);
    }
});
