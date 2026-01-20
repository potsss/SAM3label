import os
import cv2
import numpy as np
import torch
from PIL import Image
from sam3 import build_sam3_image_model, Sam3Processor
from core.converter import mask_to_polygons

class SAM3Annotator:
    def __init__(self, model_path: str):
        """
        Initialize the SAM3 model using the official sam3 library.
        """
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = build_sam3_image_model(checkpoint_path=model_path)
            self.processor = Sam3Processor(self.model)
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model.to('cuda')
                print("SAM3 model moved to GPU.")
        else:
            print(f"Warning: Model file {model_path} not found. Running in mock mode.")
            self.model = None
            self.processor = None

    def predict(self, image: np.ndarray, points=None, labels=None, boxes=None, texts=None, epsilon_ratio=0.005):
        if self.processor is None:
            # Mock return for testing without model
            print("Running in mock mode.")
            return [{"points": [[10, 10], [100, 10], [100, 100]], "label": "mock_obj"}]

        # Convert cv2 image (BGR) to PIL image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Set the image
        try:
            inference_state = self.processor.set_image(pil_image)
        except Exception as e:
            raise RuntimeError(f"Failed to set image for processor: {e}")

        # 2. Apply prompts
        # The sam3 processor seems to handle one prompt type at a time.
        # We'll prioritize text, then points, then boxes.
        output = None
        prompt_type = "unknown"

        try:
            if texts:
                prompt_type = "text"
                prompt_text = texts[0]['text']
                print(f"Applying text prompt: {prompt_text}")
                output = self.processor.set_text_prompt(state=inference_state, prompt=prompt_text)
            elif points and labels:
                prompt_type = "point"
                print(f"Applying point prompt: {points}")
                # The processor expects points in a torch tensor: [[x, y, label], ...]
                # label 1=fg, 0=bg
                point_prompts = [[p[0], p[1], l] for p, l in zip(points, labels)]
                output = self.processor.set_point_prompt(state=inference_state, points=point_prompts)
            elif boxes:
                prompt_type = "box"
                print(f"Applying box prompt: {boxes}")
                # The processor expects boxes in a list of lists: [[x1, y1, x2, y2], ...]
                output = self.processor.set_box_prompt(state=inference_state, boxes=boxes)
            else:
                # No prompts provided
                return []

            if output is None or not output.get('masks'):
                print("Model returned no output or no masks for the given prompt.")
                return []

            # 3. Process masks
            # Output contains 'masks', 'scores', 'logits'
            masks = output['masks']
            all_polygons = []
            
            for i, mask_tensor in enumerate(masks):
                # Convert torch tensor to numpy array for processing
                mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
                
                # Use the existing converter
                polys = mask_to_polygons(mask_np, epsilon_ratio=epsilon_ratio)
                
                label_name = f"{prompt_type}_{i}"
                if texts:
                    label_name = texts[0]['text']

                for p in polys:
                    all_polygons.append({
                        "points": p,
                        "label": label_name
                    })
            
            return all_polygons

        except Exception as e:
            print(f"Error during SAM3 inference with prompt type '{prompt_type}': {e}")
            # Re-raise the exception to be caught by the FastAPI error handler
            raise e