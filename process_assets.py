import sys
import timm.layers
sys.modules["timm.models.layers"] = timm.layers
import os
import torch
import numpy as np
import shutil
import cv2
from PIL import Image, ImageDraw

def load_env_vars():
    # Load HF_TOKEN from .env file
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    os.environ["HF_TOKEN"] = token
                    print("Loaded HF_TOKEN from .env")
                    break
    # Redirect Hugging Face cache
    os.environ["HF_HOME"] = os.path.abspath(".hf_cache")

def apply_monkey_patches():
    # Apply monkey patch to fix offload_state_to_cpu issue in Sam3MultiplexTrackingWithInteractivity
    try:
        from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity
        orig_init_state = Sam3MultiplexTrackingWithInteractivity.init_state
        
        def patched_init_state(self, resource_path, offload_video_to_cpu=False, offload_state_to_cpu=False, **kwargs):
            return orig_init_state(
                self,
                resource_path=resource_path,
                offload_video_to_cpu=offload_video_to_cpu,
                **kwargs
            )
            
        Sam3MultiplexTrackingWithInteractivity.init_state = patched_init_state
        print("Patched Sam3MultiplexTrackingWithInteractivity.init_state successfully.")
    except ImportError as e:
        print(f"Could not patch Sam3MultiplexTrackingWithInteractivity: {e}")

    # Patch Sam3MultiplexDetector.forward_video_grounding_batched_multigpu to fix off-by-one bug with max_frame_num_to_track
    try:
        from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
        orig_forward = Sam3MultiplexDetector.forward_video_grounding_batched_multigpu
        
        def patched_forward(self, *args, **kwargs):
            if "max_frame_num_to_track" in kwargs and kwargs["max_frame_num_to_track"] is not None:
                kwargs["max_frame_num_to_track"] += 1
            elif len(args) > 12 and args[12] is not None:
                args_list = list(args)
                args_list[12] += 1
                args = tuple(args_list)
            return orig_forward(self, *args, **kwargs)
            
        Sam3MultiplexDetector.forward_video_grounding_batched_multigpu = patched_forward
        print("Patched Sam3MultiplexDetector.forward_video_grounding_batched_multigpu successfully.")
    except Exception as e:
        print(f"Could not patch Sam3MultiplexDetector: {e}")

    # Patch Sam3MultiplexDetector.forward_video_grounding_multigpu to fix off-by-one bug with max_frame_num_to_track
    try:
        from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector
        orig_forward_single = Sam3MultiplexDetector.forward_video_grounding_multigpu
        
        def patched_forward_single(self, *args, **kwargs):
            if "max_frame_num_to_track" in kwargs and kwargs["max_frame_num_to_track"] is not None:
                kwargs["max_frame_num_to_track"] += 1
            elif len(args) > 12 and args[12] is not None:
                args_list = list(args)
                args_list[12] += 1
                args = tuple(args_list)
            return orig_forward_single(self, *args, **kwargs)
            
        Sam3MultiplexDetector.forward_video_grounding_multigpu = patched_forward_single
        print("Patched Sam3MultiplexDetector.forward_video_grounding_multigpu successfully.")
    except Exception as e:
        print(f"Could not patch Sam3MultiplexDetector single-gpu: {e}")

def save_mask_overlay(image_or_path, masks, boxes, scores, output_path, label):
    # Load original image
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path).convert("RGBA")
    else:
        img = Image.fromarray(image_or_path).convert("RGBA")
        
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    colors = [
        (255, 0, 0, 100),   # Red
        (0, 255, 0, 100),   # Green
        (0, 0, 255, 100),   # Blue
        (255, 255, 0, 100), # Yellow
        (255, 0, 255, 100), # Magenta
        (0, 255, 255, 100), # Cyan
    ]
    
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().float().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().float().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().float().numpy()
        
    if masks is not None:
        for idx, mask in enumerate(masks):
            color = colors[idx % len(colors)]
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask.astype(bool)
            
            # Create colored mask overlay
            mask_uint8 = (mask_bool * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_uint8).resize((w, h), Image.Resampling.NEAREST)
            color_img = Image.new("RGBA", (w, h), color[:3] + (0,))
            colorized = Image.composite(Image.new("RGBA", (w, h), color), color_img, mask_img)
            
            overlay = Image.alpha_composite(overlay, colorized)
            
            # Draw bounding box
            if boxes is not None and idx < len(boxes):
                box = boxes[idx]
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=3)
                
                score_str = f" {scores[idx]:.2f}" if scores is not None and idx < len(scores) else ""
                draw.text((x1 + 5, y1 + 5), f"{label}{score_str}", fill=(255, 255, 255, 255))
                
    final_img = Image.alpha_composite(img, overlay).convert("RGB")
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_img.save(output_path)
        print(f"Saved visualization to {output_path}")
    return np.array(final_img)

def test_images(processor, device):
    print("\n--- Running Image Predictor Tests ---")
    image_tests = [
        ("sam3/assets/images/test_image.jpg", "child", "results/images/test_image_child.png"),
        ("sam3/assets/images/test_image.jpg", "shoe", "results/images/test_image_shoe.png"),
        ("sam3/assets/images/truck.jpg", "truck", "results/images/truck_truck.png"),
        ("sam3/assets/images/truck.jpg", "wheel", "results/images/truck_wheel.png"),
        ("sam3/assets/images/groceries.jpg", "bottle", "results/images/groceries_bottle.png"),
        ("sam3/assets/images/groceries.jpg", "vegetables", "results/images/groceries_vegetables.png"),
    ]
    
    for img_path, prompt, out_path in image_tests:
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} (not found)")
            continue
            
        print(f"Processing image {img_path} with prompt: '{prompt}'...")
        image = Image.open(img_path)
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        save_mask_overlay(img_path, masks, boxes, scores, out_path, prompt)

def test_video_folder(predictor):
    print("\n--- Running Video Tracking Test on Frame Folder (0001) ---")
    video_dir = "sam3/assets/videos/0001"
    if not os.path.exists(video_dir):
        print(f"Skipping {video_dir} (not found)")
        return
        
    # Get frame dimensions
    first_frame_path = os.path.join(video_dir, "00000.jpg")
    if not os.path.exists(first_frame_path):
        first_frame_path = os.path.join(video_dir, "0.jpg")
    
    first_frame = Image.open(first_frame_path)
    w, h = first_frame.size

    print(f"Initializing tracking session on frame folder: {video_dir}...")
    response = predictor.handle_request({"type": "start_session", "resource_path": video_dir, "offload_video_to_cpu": True})
    session_id = response["session_id"]
    
    prompt = "person"
    print(f"Adding text prompt: '{prompt}' on frame 0...")
    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": prompt,
    })
    
    print("Propagating tracking through frame folder...")
    result_dir = "results/videos/0001_tracked"
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize MP4 VideoWriter
    mp4_path = "results/videos/0001_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(mp4_path, fourcc, 10.0, (w, h))
    
    for res in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id, "max_frame_num_to_track": 30}):
        frame_idx = res.get("frame_index")
        if frame_idx is not None:
            outputs = res.get("outputs", {})
            out_obj_ids = outputs.get("out_obj_ids", [])
            binary_masks = outputs.get("out_binary_masks")
            
            if binary_masks is not None:
                if isinstance(binary_masks, torch.Tensor):
                    binary_masks = binary_masks.cpu().numpy()
                    
                frame_path = os.path.join(video_dir, f"{frame_idx}.jpg")
                if not os.path.exists(frame_path):
                    frame_path = os.path.join(video_dir, f"{frame_idx:05d}.jpg")
                    
                if os.path.exists(frame_path):
                    png_path = os.path.join(result_dir, f"frame_{frame_idx:05d}.png") if frame_idx % 30 == 0 else None
                    overlay_rgb = save_mask_overlay(
                        frame_path,
                        binary_masks,
                        None,
                        None,
                        png_path,
                        f"obj_{out_obj_ids[0] if len(out_obj_ids) > 0 else 0}"
                    )
                    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(overlay_bgr)
                    
    video_writer.release()
    print(f"Saved compiled MP4 video to {mp4_path}")
    predictor.handle_request({"type": "close_session", "session_id": session_id})

def test_video_file(predictor):
    print("\n--- Running Video Tracking Test on Bedroom Video (bedroom.mp4) ---")
    video_path = "sam3/assets/videos/bedroom.mp4"
    if not os.path.exists(video_path):
        print(f"Skipping {video_path} (not found)")
        return
        
    print(f"Initializing tracking session on video file: {video_path}...")
    response = predictor.handle_request({"type": "start_session", "resource_path": video_path, "offload_video_to_cpu": True})
    session_id = response["session_id"]
    
    prompt = "bed"
    print(f"Adding text prompt: '{prompt}' on frame 0...")
    predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "text": prompt,
    })
    
    print("Propagating tracking through bedroom.mp4...")
    result_dir = "results/videos/bedroom_tracked"
    os.makedirs(result_dir, exist_ok=True)
    
    # We will extract frames from the MP4 using OpenCV to draw overlays on them
    cap = cv2.VideoCapture(video_path)
    frames_rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Extracted {len(frames_rgb)} reference frames from video file.")
    
    # Initialize MP4 VideoWriter
    h, w, _ = frames_rgb[0].shape
    mp4_path = "results/videos/bedroom_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(mp4_path, fourcc, 10.0, (w, h))
    
    for res in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id, "max_frame_num_to_track": 30}):
        frame_idx = res.get("frame_index")
        if frame_idx is not None and frame_idx < len(frames_rgb):
            outputs = res.get("outputs", {})
            out_obj_ids = outputs.get("out_obj_ids", [])
            binary_masks = outputs.get("out_binary_masks")
            
            if binary_masks is not None:
                if isinstance(binary_masks, torch.Tensor):
                    binary_masks = binary_masks.cpu().numpy()
                    
                # Save PNG only if frame_idx % 20 == 0
                png_path = os.path.join(result_dir, f"frame_{frame_idx:05d}.png") if frame_idx % 20 == 0 else None
                overlay_rgb = save_mask_overlay(
                    frames_rgb[frame_idx],
                    binary_masks,
                    None,
                    None,
                    png_path,
                    f"obj_{out_obj_ids[0] if len(out_obj_ids) > 0 else 0}"
                )
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                video_writer.write(overlay_bgr)
                
    video_writer.release()
    print(f"Saved compiled MP4 video to {mp4_path}")
    predictor.handle_request({"type": "close_session", "session_id": session_id})

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SAM3 Assets Test")
    parser.add_argument("--mode", choices=["image", "video", "all"], default="all", help="Execution mode")
    args = parser.parse_args()

    load_env_vars()
    apply_monkey_patches()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device} (mode: {args.mode})")
    
    from sam3.model_builder import build_sam3_image_model, build_sam3_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
    
    # Absolute BPE file path to bypass setuptools buggy package resource lookup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bpe_path = os.path.join(current_dir, "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        if args.mode in ["image", "all"]:
            # Initialize image model and run image tests
            print("Building image model...")
            image_model = build_sam3_image_model(bpe_path=bpe_path, device=device, load_from_HF=True)
            processor = Sam3Processor(image_model)
            test_images(processor, device)
            
            # Clean up image model to free GPU memory
            del processor
            del image_model
            torch.cuda.empty_cache()
            
        if args.mode in ["video", "all"]:
            # Initialize video predictor and run video tests
            print("\nBuilding video model...")
            video_predictor = build_sam3_predictor(bpe_path=bpe_path, version="sam3.1", compile=False, async_loading_frames=False, use_fa3=False)
            video_predictor.model.batched_grounding_batch_size = 2
            video_predictor.model.postprocess_batch_size = 2
            test_video_folder(video_predictor)
            test_video_file(video_predictor)
            
    print(f"\n=== Asset tests ({args.mode}) completed successfully! Overlays saved in results/ folder ===")

if __name__ == "__main__":
    main()
