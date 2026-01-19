import cv2
import numpy as np
from typing import List

def mask_to_polygons(mask: np.ndarray, epsilon_ratio: float = 0.005) -> List[List[List[float]]]:
    """
    Convert a binary mask to a list of simplified polygons.
    
    Args:
        mask: Binary mask (numpy array of 0s and 1s)
        epsilon_ratio: Simplification factor. Smaller value means more vertices.
        
    Returns:
        List of polygons, where each polygon is a list of [x, y] coordinates.
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
            
        # Simplify polygon using Douglas-Peucker algorithm
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Convert to list of [x, y]
        poly_points = approx.reshape(-1, 2).tolist()
        if len(poly_points) >= 3:
            polygons.append(poly_points)
            
    return polygons
