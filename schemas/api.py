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

# --- Models for Video Annotation (with Box Prompts) ---

class PointPrompt(BaseModel):
    point: List[float] = Field(..., description="[x, y] coordinates")
    label: int = Field(1, description="1 for positive, 0 for negative")

class VideoAnnotationRequest(BaseModel):
    video_base64: str = Field(..., description="Base64 encoded video file.")
    boxes: List[BoxPrompt] = Field(..., description="List of initial box prompts on the first frame.")

class VideoAnnotationResponse(BaseModel):
    # The keys will be frame numbers as strings, and values are the base64 encoded annotated frames
    frames: dict[str, str]
    debug_images: Optional[dict[str, str]] = None  # Keys like "frame_0", "frame_1", values are base64 strings

# --- Models for Video Tracking with Text Prompt ---

class VideoTextTrackingRequest(BaseModel):
    video_base64: str = Field(..., description="Base64 encoded video file.")
    text_prompt: str = Field(..., description="Text description of what to detect and track in the video (e.g., 'person', 'car', 'dog')")

class VideoTextTrackingResponse(BaseModel):
    frames: dict[str, str] = Field(..., description="Frame indices as keys, base64 annotated frames as values")

