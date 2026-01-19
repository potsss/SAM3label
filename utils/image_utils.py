import base64
import cv2
import numpy as np

def base64_to_cv2(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to OpenCV image (BGR).
    """
    # Remove metadata header if present (e.g., 'data:image/jpeg;base64,')
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
