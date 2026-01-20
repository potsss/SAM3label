import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
from core.converter import mask_to_polygons

class SAM3Annotator:
    def __init__(self, model_path: str):
        """
        Initialize the SAM3 model using the Hugging Face Transformers library.
        The model_path should be the directory containing the model weights and config.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if os.path.isdir(model_path):
            print(f"Loading model from directory: {model_path}")
            try:
                self.model = Sam3Model.from_pretrained(model_path).to(self.device)
                self.processor = Sam3Processor.from_pretrained(model_path)
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Falling back to mock mode.")
                self.model = None
                self.processor = None
        else:
            print(f"Warning: Model path {model_path} is not a directory. Running in mock mode.")
            self.model = None
            self.processor = None

    def predict(self, image: np.ndarray, points=None, labels=None, boxes=None, texts=None, epsilon_ratio=0.005):
        if self.processor is None:
            print("Running in mock mode.")
            return [{"points": [[10, 10], [100, 10], [100, 100]], "label": "mock_obj"}]

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        inputs = {}
        prompt_type = "unknown"

        try:
            # The processor expects a list of prompts for each type
            if texts:
                prompt_type = "text"
                inputs["text_prompts"] = [t['text'] for t in texts]
            
            if points and labels:
                prompt_type = "point"
                # The processor wants points and labels in separate lists, batched.
                # Format: [[[x1, y1], [x2, y2], ...]], [[[l1], [l2], ...]]
                inputs["input_points"] = [[[p[0], p[1]] for p in points]]
                inputs["input_labels"] = [[[l] for l in labels]]

            if boxes:
                prompt_type = "box"
                # Format: [[[x1, y1, x2, y2], ...]]
                inputs["input_boxes"] = [boxes]

            if not inputs:
                return []

            print(f"Processing with {prompt_type} prompts...")
            processed_inputs = self.processor(pil_image, **inputs, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**processed_inputs)

            # Post-process to get segmentation masks
            # Note: The target_sizes parameter is crucial for resizing masks to original image size
            original_size = [pil_image.size[::-1]] # (height, width)
            results = self.processor.post_process_instance_segmentation(
                outputs, 
                threshold=0.5, 
                mask_threshold=0.5,
                original_sizes=original_size
            )[0]
            
            masks = results['masks']
            all_polygons = []
            
            for i, mask_tensor in enumerate(masks):
                mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
                polys = mask_to_polygons(mask_np, epsilon_ratio=epsilon_ratio)
                
                label_name = f"{prompt_type}_{i}"
                if texts:
                    label_name = inputs["text_prompts"][0] # Use the first text prompt as label

                for p in polys:
                    all_polygons.append({
                        "points": p,
                        "label": label_name
                    })
            
            return all_polygons

        except Exception as e:
            print(f"Error during SAM3 inference with prompt type '{prompt_type}': {e}")
            raise e
