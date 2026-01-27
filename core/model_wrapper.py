import os
import cv2
import numpy as np
import torch
from PIL import Image
import io
import base64
import tempfile

# Imports for image and video processing
from transformers import Sam3Model, Sam3Processor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

class SAM3Annotator:
    def __init__(self, model_path: str):
        """
        Initializes the PCS model for images and the PVS Tracker model for videos.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # Load Image (PCS) Model
            print("Loading SAM3 PCS Model (for images) from local path...")
            self.pcs_model = Sam3Model.from_pretrained(model_path).to(self.device)
            self.pcs_processor = Sam3Processor.from_pretrained(model_path)
            print("PCS Model and Processor loaded successfully.")

            # Load Video (PVS Tracker) Model
            print("Loading SAM3 PVS Tracker Model (for videos) from local path...")
            self.pvs_tracker_model = Sam3TrackerVideoModel.from_pretrained(model_path).to(self.device)
            self.pvs_tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(model_path)
            print("PVS Tracker Model and Processor loaded successfully.")


        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            self.pcs_model = None
            self.pvs_tracker_model = None

    def predict(self, image: np.ndarray, boxes=None, texts=None):
        # ... (existing predict method for images - unchanged) ...
        if not self.pcs_model:
            print("Running in mock mode due to model loading failure.")
            return [{"label": "mock_obj", "mask_base64": ""}]

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
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

            output_masks = []
            for i, mask_tensor in enumerate(masks_tensor):
                mask_np_binary = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                mask_255 = mask_np_binary * 255
                blurred_mask = cv2.GaussianBlur(mask_255, (5, 5), 0)
                colored_mask = np.zeros((mask_np_binary.shape[0], mask_np_binary.shape[1], 4), dtype=np.uint8)
                color = np.array([0, 255, 255])
                colored_mask[blurred_mask > 0, :3] = color
                colored_mask[:, :, 3] = blurred_mask
                mask_img = Image.fromarray(colored_mask, 'RGBA')
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                output_masks.append({"label": f"{base_label}_{i}", "mask_base64": f"data:image/png;base64,{img_str}"})
            return output_masks
        except Exception as e:
            print(f"Error during SAM3 inference: {e}")
            raise e

    def predict_video(self, video_base64: str, boxes: list):
        if not self.pvs_tracker_model:
            raise Exception("Video model not loaded.")

        video_data = base64.b64decode(video_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(video_data)
            video_path = tmpfile.name

        try:
            # 1. Initialize video session in streaming mode (no video frames passed)
            inference_session = self.pvs_tracker_processor.init_video_session(
                inference_device=self.device
            )

            # 2. Add box prompts for the first frame
            obj_ids = list(range(1, len(boxes) + 1))
            print(f"obj_ids content: {obj_ids}")
            input_boxes = np.array(boxes).reshape(1, len(boxes), 4).tolist()
            
            # 3. Open video and process frame by frame
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file.")
            
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            video_segments = {}
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                # Process a single frame to get correct input shapes
                inputs = self.pvs_tracker_processor(images=pil_frame, return_tensors="pt")

                # On the first frame, add the user's prompts to the session
                if frame_idx == 0:
                    self.pvs_tracker_processor.add_inputs_to_inference_session(
                        inference_session=inference_session,
                        frame_idx=0,
                        obj_ids=obj_ids,
                        input_boxes=input_boxes,
                        original_size=inputs.original_sizes[0],
                    )

                # 4. Propagate model on the current frame
                with torch.no_grad():
                    model_outputs = self.pvs_tracker_model(
                        inference_session=inference_session,
                        frame=inputs.pixel_values[0].to(self.device),
                        multimask_output=False
                    )                
                # 5. Post-process the masks for the current frame
                video_res_masks_list = self.pvs_tracker_processor.post_process_masks(
                    [model_outputs.pred_masks], 
                    original_sizes=[[video_height, video_width]], 
                    binarize=False
                )
                if not video_res_masks_list:
                    video_segments[str(frame_idx)] = []
                    frame_idx += 1
                    continue
                
                video_res_masks = video_res_masks_list[0].squeeze(1)

                frame_masks = []
                for i, obj_id in enumerate(obj_ids):
                    if i >= video_res_masks.shape[0]:
                        break  # Stop if model returned fewer masks than objects

                    mask_tensor = video_res_masks[i]

                    mask_np_binary = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                    print(f"mask_np_binary sum for obj {obj_id}: {np.sum(mask_np_binary)}")
                    mask_255 = mask_np_binary * 255
                    blurred_mask = cv2.GaussianBlur(mask_255, (5, 5), 0)
                    
                    colored_mask = np.zeros((mask_np_binary.shape[0], mask_np_binary.shape[1], 4), dtype=np.uint8)
                    color_val = (i * 50) % 256
                    color = np.array([color_val, 255 - color_val, (color_val + 128) % 256])
                    colored_mask[blurred_mask > 0, :3] = color
                    colored_mask[:, :, 3] = blurred_mask

                    mask_img = Image.fromarray(colored_mask, 'RGBA')
                    buffer = io.BytesIO()
                    mask_img.save(buffer, format="PNG")
                    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                    if frame_idx == 0:
                        print(f"Frame 0, Object {obj_id}: b64_str length = {len(b64_str)}, prefix = {b64_str[:50]}...")

                    frame_masks.append({
                        "label": f"object_{obj_id}",
                        "mask_base64": f"data:image/png;base64,{b64_str}"
                    })
                video_segments[str(frame_idx)] = frame_masks
                print(f"Processed video frame {frame_idx}")
                frame_idx += 1

            cap.release()
            return video_segments

        finally:
            if os.path.exists(video_path):
                os.remove(video_path)