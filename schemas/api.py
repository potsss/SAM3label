from pydantic import BaseModel, Field
from typing import List, Optional, Union

class BoxPrompt(BaseModel):
    box: List[float] = Field(..., description="[x1, y1, x2, y2] coordinates")

class TextPrompt(BaseModel):
    text: str = Field(..., description="Text description of the object")

class AnnotationRequest(BaseModel):
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    boxes: Optional[List[BoxPrompt]] = None
    texts: Optional[List[TextPrompt]] = None

class MaskResult(BaseModel):
    label: str
    mask_base64: str # A base64 encoded PNG string of the mask

class AnnotationResponse(BaseModel):
    masks: List[MaskResult]
    message: str = "Success"
