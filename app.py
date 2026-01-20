import os
from fastapi import FastAPI, HTTPException
from schemas.api import AnnotationRequest, AnnotationResponse, Polygon
from core.model_wrapper import SAM3Annotator
from utils.image_utils import base64_to_cv2
import cv2

app = FastAPI(title="SAM3 Annotation Service")

# Initialize the annotator (Singleton)
# On the server, ensure 'sam3.pt' exists in the 'models' directory
MODEL_PATH = "models/sam3.pt"
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
    points = []
    labels = []
    if request.points:
        for p in request.points:
            points.append(p.point)
            labels.append(p.label)

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
        # Note: Depending on the exact SAM3 API in Ultralytics, 
        # parameters might need slight adjustment (e.g., passing labels)
        results = annotator.predict(
            image=image,
            points=points if points else None,
            boxes=boxes if boxes else None,
            texts=texts if texts else None,
            epsilon_ratio=request.epsilon_ratio
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # 4. Format Response
    response_polygons = [
        Polygon(points=p["points"], label=p["label"]) for p in results
    ]

    return AnnotationResponse(polygons=response_polygons)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": annotator.model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8059)
