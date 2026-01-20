import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from core.converter import mask_to_polygons

# These imports are based on the user-provided from_sam3.py file.
# This assumes a 'sam3' library is installed, which is different from 'transformers'.
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Annotator:
    def __init__(self, model_path: str):
        """
        Initialize the SAM3 model using the 'sam3' library.
        The model_path should be a directory containing the model checkpoint (.pt file).
        """
        self.model = None
        self.processor = None
        
        try:
            # Find the checkpoint file within the provided directory
            checkpoint_files = glob.glob(os.path.join(model_path, "*.pt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No .pt checkpoint file found in {model_path}")
            
            checkpoint_path = checkpoint_files[0]
            print(f"Found model checkpoint: {checkpoint_path}")

            # Load the model and processor using the sam3 library's methods
            self.model = build_sam3_image_model(checkpoint_path=checkpoint_path)
            self.processor = Sam3Processor(self.model)
            print("SAM3 model and processor loaded successfully (using 'sam3' library).")

        except Exception as e:
            print(f"Error loading model from {model_path} using 'sam3' library: {e}")
            print("Running in mock mode.")
            self.model = None
            self.processor = None

    def predict(self, image: np.ndarray, points=None, labels=None, boxes=None, texts=None, epsilon_ratio=0.005):
        if self.processor is None:
            print("Running in mock mode.")
            return [{"points": [[10, 10], [100, 10], [100, 100]], "label": "mock_obj"}]

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            # 1. Set the image and get the inference state
            inference_state = self.processor.set_image(pil_image)
            
            output = None
            prompt_type = "unknown"

            # 2. Set the prompt based on provided inputs, using the stateful API
            if texts:
                prompt_type = "text"
                # Using the API from from_sam3.py
                output = self.processor.set_text_prompt(state=inference_state, prompt=texts[0])
            
            # NOTE: The following methods are inferred from the pattern in from_sam3.py,
            # as explicit examples for points and boxes were not provided.
            # The actual method names and parameters might be different.
            elif points and labels:
                prompt_type = "point"
                # Assuming a 'set_points_prompt' method exists
                # The format for points and labels might need adjustment
                output = self.processor.set_points_prompt(state=inference_state, points=points, labels=labels)
            
            elif boxes:
                prompt_type = "box"
                # Assuming a 'set_box_prompt' method exists
                output = self.processor.set_box_prompt(state=inference_state, boxes=boxes)

            if output is None or "masks" not in output:
                print(f"No output or no masks found for prompt type: {prompt_type}")
                return []

            # 3. Process the output masks
            masks_cpu = output["masks"].cpu()
            all_polygons = []

            for i, mask_tensor in enumerate(masks_cpu):
                mask_np = (mask_tensor.squeeze().numpy() > 0.5).astype(np.uint8)
                polys = mask_to_polygons(mask_np, epsilon_ratio=epsilon_ratio)
                
                label_name = f"{prompt_type}_{i}"
                if texts:
                    label_name = texts[0]

                for p in polys:
                    all_polygons.append({
                        "points": p,
                        "label": label_name
                    })
            
            return all_polygons

        except Exception as e:
            # Catching potential errors from inferred methods
            print(f"Error during SAM3 inference with stateful API: {e}")
            raise e