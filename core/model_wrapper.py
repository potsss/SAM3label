import os
import cv2
import numpy as np
import torch
from PIL import Image

# Imports based on the new, correct documentation from Hugging Face
from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

from core.converter import mask_to_polygons

class SAM3Annotator:
    def __init__(self, model_path: str):
        """
        Initializes both the Concept Segmentation (PCS) and Visual Segmentation (PVS)
        models from the Hugging Face transformers library.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Model path is now used as a cache directory for Hugging Face models
        hf_model_name = "facebook/sam3"
        
        try:
            # 1. Load PCS Model and Processor (for text prompts)
            print("Loading SAM3 PCS Model (for text prompts)...")
            self.pcs_model = Sam3Model.from_pretrained(hf_model_name, cache_dir=model_path).to(self.device)
            self.pcs_processor = Sam3Processor.from_pretrained(hf_model_name, cache_dir=model_path)
            print("PCS Model and Processor loaded successfully.")

            # 2. Load PVS/Tracker Model and Processor (for point/box prompts)
            print("Loading SAM3 PVS/Tracker Model (for point/box prompts)...")
            self.pvs_model = Sam3TrackerModel.from_pretrained(hf_model_name, cache_dir=model_path).to(self.device)
            self.pvs_processor = Sam3TrackerProcessor.from_pretrained(hf_model_name, cache_dir=model_path)
            print("PVS/Tracker Model and Processor loaded successfully.")

        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            self.pcs_model = None
            self.pvs_model = None

    def predict(self, image: np.ndarray, points=None, labels=None, boxes=None, texts=None, epsilon_ratio=0.005):
        if not self.pcs_model or not self.pvs_model:
            print("Running in mock mode due to model loading failure.")
            return [{"points": [[10, 10], [100, 10], [100, 100]], "label": "mock_obj"}]

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            if texts:
                # --- TEXT PROMPT (PCS) ---
                print("Processing with Text Prompt (PCS Model)...")
                inputs = self.pcs_processor(images=pil_image, text=texts[0], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.pcs_model(**inputs)
                
                # Use the PCS post-processor
                results = self.pcs_processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=0.5,
                    target_sizes=[pil_image.size[::-1]]
                )[0]
                masks_tensor = results["masks"]
                label_name = texts[0]

            elif points and labels:
                # --- POINT PROMPT (PVS/Tracker) ---
                print("Processing with Point Prompt (PVS/Tracker Model)...")
                # Format the input as per the documentation: [batch, object, point, coords]
                input_points = [[points]] # e.g., [[[250, 250]]]
                input_labels = [[labels]] # e.g., [[[1]]]

                inputs = self.pvs_processor(
                    images=pil_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.pvs_model(**inputs)
                
                # Use the PVS post-processor
                masks_tensor = self.pvs_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
                label_name = f"point_{points[0][0]}_{points[0][1]}"

            elif boxes:
                # --- BOX PROMPT (PVS/Tracker) ---
                print("Processing with Box Prompt (PVS/Tracker Model)...")
                # Format the input as per the documentation: [batch, num_boxes, 4]
                input_boxes = [boxes] # e.g., [[[40, 40, 160, 160]]]

                inputs = self.pvs_processor(
                    images=pil_image, input_boxes=input_boxes, return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.pvs_model(**inputs)
                
                # Use the PVS post-processor
                masks_tensor = self.pvs_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
                label_name = f"box_{boxes[0][0]}_{boxes[0][1]}"

            else:
                return []

            # --- Convert masks to polygons ---
            all_polygons = []
            # For PVS, the output might have an extra dimension for multimask output
            if masks_tensor.dim() == 4: # [num_objects, num_masks_per_obj, H, W]
                # Take the first mask for each object
                masks_tensor = masks_tensor[:, 0, :, :]

            for mask_tensor in masks_tensor:
                mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                polys = mask_to_polygons(mask_np, epsilon_ratio=epsilon_ratio)
                
                for p in polys:
                    all_polygons.append({
                        "points": p,
                        "label": label_name
                    })
            
            return all_polygons

        except Exception as e:
            print(f"Error during SAM3 inference: {e}")
            raise e
