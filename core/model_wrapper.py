import os
import cv2
import numpy as np
from ultralytics import SAM
from core.converter import mask_to_polygons

class SAM3Annotator:
    def __init__(self, model_path: str = "sam3_b.pt"):
        """
        Initialize the SAM3 model using Ultralytics.
        On your laptop, this might fail if the file is missing, 
        but the logic will be ready for the server.
        """
        # In a real scenario, we check if the model exists
        if os.path.exists(model_path):
            self.model = SAM(model_path)
        else:
            print(f"Warning: Model file {model_path} not found. Running in mock mode.")
            self.model = None

    def predict(self, image: np.ndarray, points=None, labels=None, boxes=None, texts=None, epsilon_ratio=0.005):
        if self.model is None:
            # Mock return for testing without model
            return [{"points": [[10, 10], [100, 10], [100, 100]], "label": "mock_obj"}]

        # Ultralytics SAM3 supports:
        # points: [[x, y], ...]
        # labels: [1, 0, ...]
        # bboxes: [x1, y1, x2, y2]
        # texts: ["description1", "description2"] (Core SAM3 feature)
        
        results = self.model.predict(
            source=image,
            points=points if points else None,
            labels=labels if labels else None,
            bboxes=boxes if boxes else None,
            texts=texts if texts else None,
            device="cuda" if (hasattr(self.model, 'device') and self.model.device.type == "cuda") else "cpu"
        )

        all_polygons = []
        for result in results:
            if result.masks is not None:
                # Iterate through each mask found
                # result.names contains labels for text prompts if available
                for i, mask_data in enumerate(result.masks.data):
                    mask_np = mask_data.cpu().numpy()
                    polys = mask_to_polygons(mask_np, epsilon_ratio=epsilon_ratio)
                    
                    # Try to get a meaningful label (e.g., from text prompts)
                    label_name = f"object_{i}"
                    if hasattr(result, 'names') and i in result.names:
                        label_name = result.names[i]
                    
                    for p in polys:
                        all_polygons.append({
                            "points": p,
                            "label": label_name
                        })
        
        return all_polygons
