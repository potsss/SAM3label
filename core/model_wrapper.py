import os
import cv2
import numpy as np
import torch
from PIL import Image

# Imports based on the new, correct documentation from Hugging Face
from transformers import Sam3Model, Sam3Processor

from core.converter import mask_to_polygons

class SAM3Annotator:
    def __init__(self, model_path: str):
        """
        Initializes the Concept Segmentation (PCS) model from the 
        Hugging Face transformers library.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            print("Loading SAM3 PCS Model (for text/box prompts) from local path...")
            self.pcs_model = Sam3Model.from_pretrained(model_path).to(self.device)
            self.pcs_processor = Sam3Processor.from_pretrained(model_path)
            print("PCS Model and Processor loaded successfully.")

        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            self.pcs_model = None

    def predict(self, image: np.ndarray, boxes=None, texts=None, epsilon_ratio=0.005):
        if not self.pcs_model:
            print("Running in mock mode due to model loading failure.")
            return [{"points": [[10, 10], [100, 10], [100, 100]], "label": "mock_obj"}]

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            # --- TEXT, BOX, or MIXED PROMPT (PCS Model) ---
            print("Processing with Text/Box/Mixed Prompt (PCS Model)...")
            
            processor_args = {'images': pil_image, 'return_tensors': 'pt'}
            base_label = "prompt"
            
            if texts:
                # Use only the first text prompt.
                processor_args['text'] = texts[0]
                base_label = texts[0]
            
            if boxes:
                processor_args['input_boxes'] = [boxes] 
                box_labels = [1] * len(boxes)
                processor_args['input_boxes_labels'] = [box_labels]
                if not texts:
                    base_label = "box"

            if 'text' not in processor_args and 'input_boxes' not in processor_args:
                return []

            inputs = self.pcs_processor(**processor_args).to(self.device)
            with torch.no_grad():
                outputs = self.pcs_model(**inputs)
            
            results = self.pcs_processor.post_process_instance_segmentation(
                outputs, threshold=0.5, mask_threshold=0.5,
                target_sizes=[pil_image.size[::-1]]
            )[0]
            
            masks_tensor = results["masks"]
            
            all_polygons = []
            for i, mask_tensor in enumerate(masks_tensor):
                polys = mask_to_polygons((mask_tensor.cpu().numpy() > 0.5).astype(np.uint8), epsilon_ratio=epsilon_ratio)
                for p in polys:
                     all_polygons.append({"points": p, "label": f"{base_label}_{i}"})
            return all_polygons

        except Exception as e:
            print(f"Error during SAM3 inference: {e}")
            raise e