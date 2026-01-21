import os
import cv2
import numpy as np
import torch
from PIL import Image
import io
import base64

# Imports based on the new, correct documentation from Hugging Face
from transformers import Sam3Model, Sam3Processor

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

    def predict(self, image: np.ndarray, boxes=None, texts=None):
        if not self.pcs_model:
            print("Running in mock mode due to model loading failure.")
            return [{"label": "mock_obj", "mask_base64": ""}]

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            # --- TEXT, BOX, or MIXED PROMPT (PCS Model) ---
            print("Processing with Text/Box/Mixed Prompt (PCS Model)...")
            
            processor_args = {'images': pil_image, 'return_tensors': 'pt'}
            base_label = "prompt"
            
            is_mixed_prompt = texts and boxes
            
            if texts:
                processor_args['text'] = texts[0]
                base_label = texts[0]
            
            if boxes and not is_mixed_prompt:
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
            
            if is_mixed_prompt:
                print("Filtering results for mixed prompt...")
                filtered_masks = []
                result_boxes = results["boxes"]
                result_masks = results["masks"]

                for i, res_box in enumerate(result_boxes):
                    center_x = (res_box[0] + res_box[2]) / 2
                    center_y = (res_box[1] + res_box[3]) / 2

                    for user_box in boxes:
                        if (center_x >= user_box[0] and center_x <= user_box[2] and
                            center_y >= user_box[1] and center_y <= user_box[3]):
                            filtered_masks.append(result_masks[i])
                            break
                
                masks_tensor = torch.stack(filtered_masks) if filtered_masks else torch.empty(0)
            else:
                masks_tensor = results["masks"]

            # Convert final masks to base64 PNGs with smooth edges
            output_masks = []
            for i, mask_tensor in enumerate(masks_tensor):
                # Create a binary mask (0 or 1)
                mask_np_binary = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                
                # Convert to 0-255 grayscale and apply a Gaussian blur to soften the edges
                mask_255 = mask_np_binary * 255
                # A small kernel like (5,5) provides subtle anti-aliasing
                blurred_mask = cv2.GaussianBlur(mask_255, (5, 5), 0)
                
                # Create a colored RGBA image
                colored_mask = np.zeros((mask_np_binary.shape[0], mask_np_binary.shape[1], 4), dtype=np.uint8)
                color = np.array([0, 255, 255])  # Cyan color for the mask area
                
                # Apply the color to all pixels that will be visible
                colored_mask[blurred_mask > 0, :3] = color
                # Use the blurred, smoothed mask as the alpha channel
                colored_mask[:, :, 3] = blurred_mask

                mask_img = Image.fromarray(colored_mask, 'RGBA')
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                output_masks.append({
                    "label": f"{base_label}_{i}",
                    "mask_base64": f"data:image/png;base64,{img_str}"
                })
            return output_masks

        except Exception as e:
            print(f"Error during SAM3 inference: {e}")
            raise e