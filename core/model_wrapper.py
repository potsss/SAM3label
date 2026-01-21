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
            # 1. Load PCS Model and Processor (for text prompts) from the local directory
            print("Loading SAM3 PCS Model (for text prompts) from local path...")
            self.pcs_model = Sam3Model.from_pretrained(model_path).to(self.device)
            self.pcs_processor = Sam3Processor.from_pretrained(model_path)
            print("PCS Model and Processor loaded successfully.")

            # 2. Load PVS/Tracker Model and Processor (for point/box prompts) from the local directory
            print("Loading SAM3 PVS/Tracker Model (for point/box prompts) from local path...")
            self.pvs_model = Sam3TrackerModel.from_pretrained(model_path).to(self.device)
            self.pvs_processor = Sam3TrackerProcessor.from_pretrained(model_path)
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
            # PVS model is only for point-based prompts (or box-only prompts without text).
            # PCS model handles text, boxes, and mixed prompts.
            # We will route to PVS only if points are the *only* prompt.
            # Otherwise, we use the more versatile PCS model.
            
            is_point_only_prompt = points and not texts

            if is_point_only_prompt:
                # --- POINT PROMPT (PVS/Tracker Model) ---
                print("Processing with Point Prompt (PVS/Tracker Model)...")
                input_points = [[ [p] for p in points ]]
                input_labels = [[ [l] for l in labels ]]

                inputs = self.pvs_processor(
                    images=pil_image, input_points=input_points, input_labels=input_labels, return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.pvs_model(**inputs, multimask_output=False)

                post_process_args = (outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())
                if "reshaped_input_sizes" in inputs:
                    post_process_args += (inputs["reshaped_input_sizes"].cpu(),)
                
                masks_tensor = self.pvs_processor.post_process_masks(*post_process_args)[0]
                
                # Labeling for point prompts
                all_polygons = []
                if masks_tensor.dim() == 4:
                    masks_tensor = masks_tensor[:, 0, :, :]
                for i, mask_tensor in enumerate(masks_tensor):
                    polys = mask_to_polygons((mask_tensor.cpu().numpy() > 0.5).astype(np.uint8), epsilon_ratio=epsilon_ratio)
                    for p in polys:
                        all_polygons.append({"points": p, "label": f"point_{i}"})
                return all_polygons

            else:
                # --- TEXT, BOX, or MIXED PROMPT (PCS Model) ---
                print("Processing with Text/Box/Mixed Prompt (PCS Model)...")
                
                processor_args = {'images': pil_image, 'return_tensors': 'pt'}
                base_label = "prompt"
                
                if texts:
                    # Use only the first text prompt, even if multiple are sent.
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
