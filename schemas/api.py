from pydantic import BaseModel, Field
from typing import List, Optional, Union

class PointPrompt(BaseModel):
    point: List[float] = Field(..., description="[x, y] coordinates")
    label: int = Field(1, description="1 for positive, 0 for negative")

class BoxPrompt(BaseModel):
    box: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates")

class TextPrompt(BaseModel):
    text: str = Field(..., description="Text description of the object")

class AnnotationRequest(BaseModel):
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    points: Optional[List[PointPrompt]] = None
    boxes: Optional[List[BoxPrompt]] = None
    texts: Optional[List[TextPrompt]] = None
    # Threshold for polygon simplification
    epsilon_ratio: float = Field(0.005, description="Ratio for Douglas-Peucker simplification. Smaller = more detail.")

class Polygon(BaseModel):
    points: List[List[float]] # [[x1, y1], [x2, y2], ...]
    label: str

class AnnotationResponse(BaseModel):
    polygons: List[Polygon]
    message: str = "Success"
