import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas.api import (
    AnnotationRequest, AnnotationResponse, MaskResult, 
    VideoAnnotationRequest, VideoAnnotationResponse,
    VideoTextTrackingRequest, VideoTextTrackingResponse
)
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

    # 3. Format Response
    return VideoAnnotationResponse(frames=frame_results)

@app.post("/predict_video_text", response_model=VideoTextTrackingResponse)
async def predict_video_with_text(request: VideoTextTrackingRequest):
    """
    Track and segment specific objects in video using text description.
    The model will detect and track all instances matching the text prompt across frames.
    """
    if not request.text_prompt or request.text_prompt.strip() == "":
        raise HTTPException(status_code=400, detail="Text prompt is required for video tracking.")

    # Inference with text prompt
    try:
        frame_results = annotator.predict_video_with_text(
            video_base64=request.video_base64,
            text_prompt=request.text_prompt
        )
    except Exception as e:
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Video text tracking error: {str(e)}\n\nTraceback:\n{tb_str}")

    # Format Response
    return VideoTextTrackingResponse(frames=frame_results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8069)
