import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas.api import AnnotationRequest, AnnotationResponse, MaskResult, VideoAnnotationRequest, VideoAnnotationResponse
from core.model_wrapper import SAM3Annotator
from utils.image_utils import base64_to_cv2
import cv2
import json
import base64

app = FastAPI(title="SAM3 Annotation Service")

# Add CORS middleware to allow requests from the web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize the annotator (Singleton)
# On the server, ensure the 'sam3_model' directory exists
MODEL_PATH = "sam3/"
annotator = SAM3Annotator(model_path=MODEL_PATH)

@app.post("/predict", response_model=AnnotationResponse)
async def predict_annotation(request: AnnotationRequest):
    # 1. Load Image
    if request.image_base64:
        try:
            image = base64_to_cv2(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
    elif request.image_path:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Image path not found")
        image = cv2.imread(request.image_path)
    else:
        raise HTTPException(status_code=400, detail="Either image_base64 or image_path must be provided")

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    # 2. Prepare Prompts
    boxes = []
    if request.boxes:
        for b in request.boxes:
            boxes.append(b.box)

    texts = []
    if request.texts:
        for t in request.texts:
            texts.append(t.text)

    # 3. Inference
    try:
        # The predict function now returns a list of dictionaries with mask and label
        mask_results = annotator.predict(
            image=image,
            boxes=boxes if boxes else None,
            texts=texts if texts else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # 4. Format Response
    return AnnotationResponse(masks=mask_results)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": annotator.model is not None}

@app.post("/predict_video", response_model=VideoAnnotationResponse)
async def predict_video(request: VideoAnnotationRequest):
    # 1. Prepare Prompts from the request
    boxes = []
    if request.boxes:
        for b in request.boxes:
            boxes.append(b.box)
    
    if not boxes:
        raise HTTPException(status_code=400, detail="Initial box prompts are required for video tracking.")

    # 2. Inference
    try:
        frame_results = annotator.predict_video(
            video_base64=request.video_base64,
            boxes=boxes,
        )
    except Exception as e:
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Video inference error: {str(e)}\n\nTraceback:\n{tb_str}")

    debug_image_b64 = None # Initialize
    # --- DEBUG: Save first frame as image ---
    if "0" in frame_results:
        try:
            print("[DEBUG] Saving frame 0 to debug_frame_0.jpg")
            # The string is "data:image/jpeg;base64,..." - we need to split it
            # NOTE: We now get the raw b64_str from here, before prepending the data URI header
            header, raw_b64_str = frame_results["0"].split(",", 1)
            img_data = base64.b64decode(raw_b64_str)
            with open("debug_frame_0.jpg", "wb") as f:
                f.write(img_data)
            print("[DEBUG] Successfully saved debug_frame_0.jpg")
            debug_image_b64 = f"data:image/jpeg;base64,{raw_b64_str}" # Store with prefix for response
        except Exception as e:
            print(f"[DEBUG] FAILED to save debug frame: {e}")
    # --- END DEBUG ---

    # 4. Format Response
    # Prepend data URI header to each base64 string
    prefixed_frame_results = {}
    for frame_idx, b64_string in frame_results.items():
        prefixed_frame_results[frame_idx] = f"data:image/jpeg;base64,{b64_string}"
    return VideoAnnotationResponse(frames=prefixed_frame_results, debug_image_base64=debug_image_b64)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8069)
