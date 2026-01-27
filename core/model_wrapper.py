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

        print("\n[DEBUG] Starting video prediction...")
        video_data = base64.b64decode(video_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(video_data)
            video_path = tmpfile.name

        try:
            print("[DEBUG] Initializing PVS tracker video session.")
            inference_session = self.pvs_tracker_processor.init_video_session(
                inference_device=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            )

            obj_ids = list(range(1, len(boxes) + 1))
            input_boxes = np.array(boxes).reshape(1, len(boxes), 4).tolist()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file.")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[DEBUG] Video opened successfully. Total frames: {total_frames}")

            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            annotated_frames = {}
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[DEBUG] End of video stream.")
                    break
                
                print(f"\n--- Processing frame {frame_idx}/{total_frames-1} ---")

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.pvs_tracker_processor(images=pil_frame, return_tensors="pt")

                if frame_idx == 0:
                    print("[DEBUG] Adding initial box prompts to the session for frame 0.")
                    self.pvs_tracker_processor.add_inputs_to_inference_session(
                        inference_session=inference_session,
                        frame_idx=0,
                        obj_ids=obj_ids,
                        input_boxes=input_boxes,
                        original_size=inputs.original_sizes[0],
                    )

                with torch.no_grad():
                    model_outputs = self.pvs_tracker_model(
                        inference_session=inference_session,
                        frame=inputs.pixel_values[0].to(self.device),
                        multimask_output=False
                    )                
                
                video_res_masks_list = self.pvs_tracker_processor.post_process_masks(
                    [model_outputs.pred_masks], 
                    original_sizes=[[video_height, video_width]], 
                    binarize=False
                )

                final_frame_pil = pil_frame.convert("RGBA")

                if video_res_masks_list:
                    video_res_masks = video_res_masks_list[0].squeeze(1)
                    print(f"[DEBUG] Found {video_res_masks.shape[0]} masks for frame {frame_idx}.")
                    
                    combined_overlay = Image.new("RGBA", final_frame_pil.size, (0, 0, 0, 0))

                    for i, obj_id in enumerate(obj_ids):
                        if i >= video_res_masks.shape[0]:
                            print(f"[DEBUG] Warning: Model returned fewer masks than tracked objects. Stopping at mask {i}.")
                            break

                        mask_tensor = video_res_masks[i]
                        mask_np_binary = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                        
                        color_val = (i * 60) % 256
                        color = (color_val, 255 - color_val, (color_val + 128) % 256, 150)
                        
                        mask_for_drawing = Image.fromarray(mask_np_binary * 255, 'L')
                        combined_overlay.paste(color, (0,0), mask_for_drawing)

                    final_frame_pil = Image.alpha_composite(final_frame_pil, combined_overlay)
                    print(f"[DEBUG] Successfully composited masks onto frame {frame_idx}.")
                else:
                    print(f"[DEBUG] No masks found for frame {frame_idx}. Frame will be unannotated.")

                buffer = io.BytesIO()
                final_frame_pil.convert("RGB").save(buffer, format="JPEG")
                b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                print(f"[DEBUG] Encoded frame {frame_idx} to base64 string (length: {len(b64_str)}).")
                annotated_frames[str(frame_idx)] = f"data:image/jpeg;base64,{b64_str}"
                frame_idx += 1

            cap.release()
            print(f"\n[DEBUG] Finished processing. Returning {len(annotated_frames)} annotated frames.")
            return annotated_frames

        finally:
            if os.path.exists(video_path):
                print(f"[DEBUG] Deleting temporary video file: {video_path}")
                os.remove(video_path)