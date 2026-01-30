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
            
            print("All models loaded successfully. Automatic mask generation will use the PCS model.")

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
        Uses a grid-based point sampling approach with SAM3 Tracker model.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries with mask_base64 for each detected object
        """
        # Note: SAM3 PCS model only supports text and box prompts, not point prompts.
        # For automatic grid-based point sampling, we use Sam3Tracker model instead.
        # However, Sam3Tracker is designed for single-object interactive segmentation.
        # For now, we'll use a text-based approach with a generic "object" prompt
        # to detect all objects in the image.
        
        if not self.pcs_model:
            raise Exception("PCS model not loaded.")
        
        print("\n[AUTO] Starting automatic mask generation for image...")
        print("[AUTO] Using PCS model with text-based detection...")
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        try:
            # Use a generic text prompt to detect all objects
            # SAM3 PCS model will find all instances matching this concept
            generic_prompts = ["object", "thing", "item"]
            all_masks = []
            detected_mask_hashes = set()
            
            for prompt_text in generic_prompts:
                print(f"[AUTO] Attempting detection with text prompt: '{prompt_text}'...")
                
                try:
                    # Process with PCS model using text prompt
                    inputs = self.pcs_processor(
                        images=pil_image,
                        text=prompt_text,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.pcs_model(**inputs)
                    
                    results = self.pcs_processor.post_process_instance_segmentation(
                        outputs, threshold=0.5, mask_threshold=0.5,
                        target_sizes=[pil_image.size[::-1]]
                    )[0]
                    
                    if "masks" in results and len(results["masks"]) > 0:
                        masks_found = len(results["masks"])
                        print(f"[AUTO] Found {masks_found} masks with prompt '{prompt_text}'")
                        
                        # De-duplicate masks using hash
                        for mask_tensor in results["masks"]:
                            mask_np = mask_tensor.cpu().numpy()
                            # Create hash from mask to detect duplicates across prompts
                            mask_hash = hash(tuple(mask_np.flatten()[:100]))
                            
                            if mask_hash not in detected_mask_hashes:
                                detected_mask_hashes.add(mask_hash)
                                all_masks.append(mask_tensor)
                
                except Exception as e:
                    print(f"[AUTO] Text prompt '{prompt_text}' failed: {str(e)[:80]}")
                    continue
            
            print(f"[AUTO] Detected {len(all_masks)} unique masks")
            
            output_masks = []
            for i, mask_tensor in enumerate(all_masks):
                mask_np = mask_tensor.cpu().numpy()
                mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                
                # Create RGBA mask image
                colored_mask = np.zeros((mask_binary.shape[-2], mask_binary.shape[-1], 4), dtype=np.uint8)
                color = np.array([0, 255, 255])  # Yellow for auto-detected objects
                
                mask_2d = mask_binary[0] if mask_binary.ndim == 3 else mask_binary
                colored_mask[mask_2d > 0, :3] = color
                colored_mask[:, :, 3] = mask_2d
                
                mask_img = Image.fromarray(colored_mask, 'RGBA')
                buffer = io.BytesIO()
                mask_img.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                output_masks.append({
                    "label": f"auto_object_{i}",
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
        Uses text-based prompts with SAM3 PCS model on each frame.
        
        Args:
            video_base64: Base64 encoded video file
            
        Returns:
            Dictionary with frame indices as keys and base64 annotated frames as values
        """
        if not self.pcs_model:
            raise Exception("PCS model not loaded.")
        
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
            generic_prompts = ["object"]  # Use simpler prompts for video (faster)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[AUTO] End of video stream.")
                    break
                
                print(f"--- Processing frame {frame_idx}/{total_frames-1} ---")
                
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_cv = frame.copy()
                
                # Collect masks from text prompts
                all_masks = []
                detected_mask_hashes = set()
                
                for prompt_text in generic_prompts:
                    try:
                        inputs = self.pcs_processor(
                            images=pil_frame,
                            text=prompt_text,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.pcs_model(**inputs)
                        
                        results = self.pcs_processor.post_process_instance_segmentation(
                            outputs, threshold=0.5, mask_threshold=0.5,
                            target_sizes=[pil_frame.size[::-1]]
                        )[0]
                        
                        if "masks" in results and len(results["masks"]) > 0:
                            for mask_tensor in results["masks"]:
                                mask_np = mask_tensor.cpu().numpy()
                                mask_hash = hash(tuple(mask_np.flatten()[:100]))
                                
                                if mask_hash not in detected_mask_hashes:
                                    detected_mask_hashes.add(mask_hash)
                                    all_masks.append(mask_tensor)
                    except Exception as e:
                        print(f"[AUTO] Text prompt '{prompt_text}' failed in frame {frame_idx}: {str(e)[:60]}")
                        continue
                
                print(f"[AUTO] Found {len(all_masks)} masks in frame {frame_idx}")
                
                # Apply masks to frame with colors
                if all_masks:
                    n_masks = len(all_masks)
                    colors_bgr = [
                        ((i * 60) % 256, (255 - (i * 60) % 256), ((i * 60 + 128) % 256))
                        for i in range(n_masks)
                    ]
                    
                    for i, mask_tensor in enumerate(all_masks):
                        mask_np = mask_tensor.cpu().numpy()
                        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255
                        mask_2d = mask_binary[0] if mask_binary.ndim == 3 else mask_binary
                        
                        color_bgr = colors_bgr[i]
                        colored_overlay = np.zeros_like(frame_cv, dtype=np.uint8)
                        colored_overlay[:, :] = color_bgr
                        
                        alpha = mask_2d.astype(float) / 255.0 * 0.6
                        
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