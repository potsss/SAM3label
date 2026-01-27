import os
import cv2
import numpy as np
import torch
from PIL import Image
import io
import base64
import tempfile
import gc

# Imports for image and video processing
from transformers import Sam3Model, Sam3Processor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, pipeline

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
            
            # Load Automatic Mask Generation - Try local path first, then fallback
            print("Loading SAM3 Automatic Mask Generation pipeline...")
            try:
                # Try to load from local path
                self.mask_generator = pipeline("mask-generation", model=model_path, device=0 if self.device == "cuda" else -1)
                print("Automatic Mask Generation pipeline loaded from local path successfully.")
            except Exception as e:
                print(f"Warning: Could not load automatic mask generation pipeline from local path: {e}")
                print("Automatic mask generation will be disabled. Manual annotation modes will still work.")
                self.mask_generator = None

        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            self.pcs_model = None
            self.pvs_tracker_model = None
            self.mask_generator = None

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
        """
        Process video with streaming approach to avoid CUDA memory overflow for long videos.
        Key optimizations:
        1. Stream-based processing - don't load all frames at once
        2. Explicit garbage collection after each frame
        3. GPU cache clearing to prevent memory accumulation
        4. Process and store only essential data
        """
        if not self.pvs_tracker_model:
            raise Exception("Video model not loaded.")

        print("\n[OPTIMIZE] Starting optimized video prediction with memory management...")
        video_data = base64.b64decode(video_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(video_data)
            video_path = tmpfile.name

        try:
            print("[OPTIMIZE] Initializing PVS tracker video session.")
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
            print(f"[OPTIMIZE] Video opened. Total frames: {total_frames}")

            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            annotated_frames = {}
            frame_idx = 0
            
            # Key optimization: Track memory and clear periodically
            MEMORY_CLEAR_INTERVAL = 10  # Clear GPU memory every N frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[OPTIMIZE] End of video stream.")
                    break
                
                print(f"--- Processing frame {frame_idx}/{total_frames-1} ---")

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.pvs_tracker_processor(images=pil_frame, return_tensors="pt")

                if frame_idx == 0:
                    print("[OPTIMIZE] Adding initial box prompts for frame 0.")
                    self.pvs_tracker_processor.add_inputs_to_inference_session(
                        inference_session=inference_session,
                        frame_idx=0,
                        obj_ids=obj_ids,
                        input_boxes=input_boxes,
                        original_size=inputs.original_sizes[0],
                    )

                # Process frame with GPU
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

                # Process and encode frame
                if video_res_masks_list:
                    video_res_masks = video_res_masks_list[0].squeeze(1)
                    print(f"[OPTIMIZE] Found {video_res_masks.shape[0]} masks for frame {frame_idx}.")
                    
                    # Get raw mask values and apply threshold
                    masks_raw = video_res_masks.cpu().numpy()
                    masks_binary = (masks_raw > 0.5).astype(np.uint8) * 255
                    
                    # Create annotated frame
                    frame_cv = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                    tracking_obj_ids = inference_session.obj_ids
                    
                    # Define colors for each object
                    n_masks = masks_binary.shape[0]
                    colors_bgr = [
                        ((i * 60) % 256, (255 - (i * 60) % 256), ((i * 60 + 128) % 256))
                        for i in range(n_masks)
                    ]

                    # Apply masks with color blending
                    for i, obj_id in enumerate(tracking_obj_ids):
                        if i >= masks_binary.shape[0]:
                            break
                        
                        mask_np = masks_binary[i]
                        color_bgr = colors_bgr[i]
                        colored_overlay = np.zeros_like(frame_cv, dtype=np.uint8)
                        colored_overlay[:, :] = color_bgr
                        
                        alpha = mask_np.astype(float) / 255.0 * 0.6
                        
                        for c in range(3):
                            frame_cv[:, :, c] = (
                                frame_cv[:, :, c] * (1 - alpha) + 
                                colored_overlay[:, :, c] * alpha
                            ).astype(np.uint8)
                    
                    frame_cv_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                    final_frame_pil = Image.fromarray(frame_cv_rgb, 'RGB')
                else:
                    # No masks found, use original frame
                    final_frame_pil = pil_frame

                # Encode frame to base64
                buffer = io.BytesIO()
                final_frame_pil.convert("RGB").save(buffer, format="JPEG", quality=90)
                b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                print(f"[OPTIMIZE] Encoded frame {frame_idx} to base64 (size: {len(b64_str)} bytes).")
                annotated_frames[str(frame_idx)] = f"data:image/jpeg;base64,{b64_str}"
                
                # Key optimization: Periodic memory cleanup
                if frame_idx > 0 and frame_idx % MEMORY_CLEAR_INTERVAL == 0:
                    print(f"[OPTIMIZE] Clearing GPU memory at frame {frame_idx}...")
                    
                    # Clear PyTorch GPU cache
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Explicit garbage collection
                    gc.collect()
                    
                    if self.device == "cuda":
                        print(f"[OPTIMIZE] GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                # Clear intermediate tensors
                del model_outputs, inputs, video_res_masks_list
                if 'masks_raw' in locals():
                    del masks_raw
                if 'masks_binary' in locals():
                    del masks_binary
                
                frame_idx += 1

            cap.release()
            print(f"\n[OPTIMIZE] Finished processing. Total frames returned: {len(annotated_frames)}")
            
            # Final cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return annotated_frames

        finally:
            if os.path.exists(video_path):
                print(f"[OPTIMIZE] Deleting temporary video file: {video_path}")
                os.remove(video_path)
            
            # Ensure cleanup on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def predict_auto_image(self, image: np.ndarray):
        """
        Automatically generate masks for all objects in an image without any prompts.
        Uses SAM3's automatic mask generation feature.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries with mask_base64 for each detected object
        """
        if not self.mask_generator:
            raise Exception(
                "Automatic mask generator not available. "
                "This feature requires internet connection to download the pipeline from HuggingFace, "
                "or a locally cached model. "
                "Please use manual annotation mode (text or box prompts) instead."
            )
        
        print("\n[AUTO] Starting automatic mask generation for image...")
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            # Use the automatic mask generation pipeline
            # points_per_batch controls the density of exploration points
            outputs = self.mask_generator(pil_image, points_per_batch=64)
            
            print(f"[AUTO] Detected {len(outputs['masks'])} masks")
            
            output_masks = []
            
            for i, mask in enumerate(outputs['masks']):
                # Convert mask to numpy array if needed
                if hasattr(mask, 'numpy'):
                    mask_np = mask.numpy()
                else:
                    mask_np = np.array(mask)
                
                # Ensure binary mask
                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                
                # Create RGBA mask image
                colored_mask = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 4), dtype=np.uint8)
                color = np.array([0, 255, 255])  # Yellow for auto-detected objects
                colored_mask[mask_binary > 0, :3] = color
                colored_mask[:, :, 3] = mask_binary
                
                mask_img = Image.fromarray(colored_mask, 'RGBA')
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                output_masks.append({
                    "label": f"object_{i}",
                    "mask_base64": f"data:image/png;base64,{img_str}"
                })
            
            print(f"[AUTO] Successfully generated {len(output_masks)} masks")
            return output_masks
            
        except Exception as e:
            print(f"Error during automatic mask generation: {e}")
            raise e
    
    def predict_auto_video(self, video_base64: str):
        """
        Automatically generate and segment masks for all frames in a video.
        Uses SAM3's automatic mask generation feature on each frame.
        
        Args:
            video_base64: Base64 encoded video file
            
        Returns:
            Dictionary with frame indices as keys and base64 annotated frames as values
        """
        if not self.mask_generator:
            raise Exception(
                "Automatic mask generator not available. "
                "This feature requires internet connection to download the pipeline from HuggingFace, "
                "or a locally cached model. "
                "Please use manual annotation mode (text or box prompts) instead."
            )
        
        print("\n[AUTO] Starting automatic video segmentation...")
        video_data = base64.b64decode(video_base64)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
            tmpfile.write(video_data)
            video_path = tmpfile.name

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file.")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            print(f"[AUTO] Video info: {video_width}x{video_height}, Total frames: {total_frames}")
            
            annotated_frames = {}
            frame_idx = 0
            MEMORY_CLEAR_INTERVAL = 5  # Clear GPU memory every N frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[AUTO] End of video stream.")
                    break
                
                print(f"--- Processing frame {frame_idx}/{total_frames-1} ---")
                
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Generate masks automatically
                outputs = self.mask_generator(pil_frame, points_per_batch=64)
                masks = outputs['masks']
                
                print(f"[AUTO] Found {len(masks)} objects in frame {frame_idx}")
                
                # Create annotated frame
                frame_cv = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                
                # Define colors for each detected object
                n_masks = len(masks)
                colors_bgr = [
                    ((i * 60) % 256, (255 - (i * 60) % 256), ((i * 60 + 128) % 256))
                    for i in range(n_masks)
                ]
                
                # Apply masks with color blending
                for i, mask in enumerate(masks):
                    if hasattr(mask, 'numpy'):
                        mask_np = mask.numpy()
                    else:
                        mask_np = np.array(mask)
                    
                    mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                    color_bgr = colors_bgr[i]
                    
                    # Create colored overlay
                    colored_overlay = np.zeros_like(frame_cv, dtype=np.uint8)
                    colored_overlay[:, :] = color_bgr
                    
                    # Blend
                    alpha = mask_binary.astype(float) / 255.0 * 0.6
                    
                    for c in range(3):
                        frame_cv[:, :, c] = (
                            frame_cv[:, :, c] * (1 - alpha) + 
                            colored_overlay[:, :, c] * alpha
                        ).astype(np.uint8)
                
                # Encode frame to base64
                frame_cv_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
                final_frame_pil = Image.fromarray(frame_cv_rgb, 'RGB')
                
                buffer = io.BytesIO()
                final_frame_pil.convert("RGB").save(buffer, format="JPEG", quality=90)
                b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                annotated_frames[str(frame_idx)] = f"data:image/jpeg;base64,{b64_str}"
                
                # Periodic memory cleanup
                if frame_idx > 0 and frame_idx % MEMORY_CLEAR_INTERVAL == 0:
                    print(f"[AUTO] Clearing GPU memory at frame {frame_idx}...")
                    
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    gc.collect()
                
                frame_idx += 1

            cap.release()
            print(f"\n[AUTO] Finished processing. Total frames returned: {len(annotated_frames)}")
            
            # Final cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return annotated_frames

        finally:
            if os.path.exists(video_path):
                print(f"[AUTO] Deleting temporary video file: {video_path}")
                os.remove(video_path)
            
            # Ensure cleanup on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            # Ensure cleanup on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()