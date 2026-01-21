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
            
            # The model does not natively support "find text in box".
            # The workaround is to find all instances of the text, then filter them
            # to see which ones are inside the user-provided boxes.
            
            # For mixed prompts, we ignore the boxes during the initial API call.
            is_mixed_prompt = texts and boxes
            
            if texts:
                processor_args['text'] = texts[0]
                base_label = texts[0]
            
            if boxes and not is_mixed_prompt:
                # Only add boxes to the processor if it's a box-only prompt
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
            
            # Post-processing for mixed prompts (text + box)
            if is_mixed_prompt:
                print("Filtering results for mixed prompt...")
                filtered_masks = []
                result_boxes = results["boxes"]
                result_masks = results["masks"]

                for i, res_box in enumerate(result_boxes):
                    # Calculate center of the mask's bounding box
                    center_x = (res_box[0] + res_box[2]) / 2
                    center_y = (res_box[1] + res_box[3]) / 2

                    # Check if the center is inside any of the user-provided prompt boxes
                    for user_box in boxes:
                        if (center_x >= user_box[0] and center_x <= user_box[2] and
                            center_y >= user_box[1] and center_y <= user_box[3]):
                            filtered_masks.append(result_masks[i])
                            break # Found a match, no need to check other user boxes
                
                masks_tensor = torch.stack(filtered_masks) if filtered_masks else torch.empty(0)
            else:
                masks_tensor = results["masks"]

            # Convert final masks to polygons
            all_polygons = []
            for i, mask_tensor in enumerate(masks_tensor):
                polys = mask_to_polygons((mask_tensor.cpu().numpy() > 0.5).astype(np.uint8), epsilon_ratio=epsilon_ratio)
                for p in polys:
                     all_polygons.append({"points": p, "label": f"{base_label}_{i}"})
            return all_polygons

        except Exception as e:
            print(f"Error during SAM3 inference: {e}")
            raise e