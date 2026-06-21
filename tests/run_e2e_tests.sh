#!/bin/bash
# run_e2e_tests.sh - Unified SAM3 End-to-End Test Script

# Exit immediately if a command exits with a non-zero status
set -e

# Resolve script directory and change to workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Setup cleanup trap for temporary Python test script
TEMP_TEST_FILE="tests/temp_test_sam3_e2e.py"
cleanup() {
    if [ -f "$TEMP_TEST_FILE" ]; then
        echo "Cleaning up temporary test file..."
        rm -f "$TEMP_TEST_FILE"
    fi
}
trap cleanup EXIT

echo "=== Setting up SAM3 End-to-End Environment ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' command is not installed or not in PATH." >&2
    exit 1
fi

# Set environmental variables
export UV_HTTP_TIMEOUT=300
export HF_HOME="./.hf_cache"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

# Install the local sam3 package as an editable dependency with notebook & dev extras
echo "Installing/updating local sam3 submodule and its optional dependencies..."
uv add --index-strategy unsafe-best-match --editable "./sam3[notebooks,dev]"

echo ""
echo "========================================="
echo "Running SAM3 Unit/Feature Tests (Inline)"
echo "========================================="

# Write Python E2E unit test code to a temporary file
cat << 'EOF' > "$TEMP_TEST_FILE"
import sys
import timm.layers
sys.modules["timm.models.layers"] = timm.layers

import os
import torch
import numpy as np
import shutil
from PIL import Image, ImageDraw

def load_env_vars():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    os.environ["HF_TOKEN"] = token
                    print("Loaded HF_TOKEN from .env")
                    break

def create_sample_image(path):
    # Create a 512x512 image: a red square in the middle on a black background
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Draw a red square
    img[128:384, 128:384, 0] = 255
    pil_img = Image.fromarray(img)
    pil_img.save(path)
    print(f"Created sample image at {path}")

def create_synthetic_frames(directory, n_frames=5):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    # Generate simple frames of a moving circle
    for idx in range(n_frames):
        img = Image.new("RGB", (256, 256), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Circle moving from left to right
        x = 50 + idx * 20
        y = 128
        draw.ellipse([(x - 20, y - 20), (x + 20, y + 20)], fill=(0, 255, 0)) # Green circle
        img.save(os.path.join(directory, f"{idx:05d}.jpg"))
    print(f"Generated {n_frames} synthetic video frames in {directory}")

def test_image_model():
    print("\n=== Running SAM3 Image Feature Test ===")
    
    # Import model builders
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    # Create visual/text model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM3 image model on {device}...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    bpe_path = os.path.join(parent_dir, "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, device=device, load_from_HF=True)
    processor = Sam3Processor(model)
    
    # Load sample image
    image_path = "sample_image.png"
    create_sample_image(image_path)
    try:
        image = Image.open(image_path)
        
        # Start inference
        print("Running inference...")
        inference_state = processor.set_image(image)
        
        # Prompt with concept "red square"
        prompt = "red square"
        print(f"Prompting model with: '{prompt}'")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        print(f"Inference output:")
        print(f"  Masks: found {len(masks)} mask(s)")
        if len(masks) > 0:
            print(f"  Boxes: {boxes}")
            print(f"  Scores: {scores}")
        
        assert len(masks) > 0, "No masks were returned!"
        print("SAM3 Image Model Test SUCCESSFUL!")
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Removed temporary sample image: {image_path}")

def save_mask_overlay(image, masks, boxes, scores, output_path, label):
    img = image.convert("RGBA")
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

def test_example_image_prompts():
    print("\n=== Running SAM3 Example Image Prompts Test ===")
    
    # Import model builders
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM3 image model on {device}...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    bpe_path = os.path.join(parent_dir, "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, device=device, load_from_HF=True)
    processor = Sam3Processor(model)
    
    image_prompts = {
        "sam3/assets/images/test_image.jpg": [
            "shoe",
            "the leftmost child wearing blue vest",
            "child"
        ],
        "sam3/assets/images/truck.jpg": [
            "truck",
            "wheel"
        ],
        "sam3/assets/images/groceries.jpg": [
            "bag",
            "food"
        ]
    }
    
    for relative_path, prompts in image_prompts.items():
        image_path = os.path.join(parent_dir, relative_path)
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}. Skipping.")
            continue
            
        print(f"\nProcessing {relative_path}...")
        image = Image.open(image_path)
        inference_state = processor.set_image(image)
        
        for prompt in prompts:
            # Clean/reset prompts first to avoid conflict in state
            processor.reset_all_prompts(inference_state)
            print(f"  Prompting with: '{prompt}'...")
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            print(f"    Found {len(masks)} mask(s) (Scores: {scores.tolist() if hasattr(scores, 'tolist') else scores})")
            assert len(masks) > 0, f"No masks returned for {relative_path} with prompt '{prompt}'!"
            
            # Save the result to a different image file
            img_basename = os.path.basename(relative_path).split(".")[0]
            prompt_sanitized = prompt.replace(" ", "_")
            out_dir = os.path.join(parent_dir, "results/images")
            output_path = os.path.join(out_dir, f"{img_basename}_{prompt_sanitized}.png")
            save_mask_overlay(image, masks, boxes, scores, output_path, prompt)
            
    print("\nSAM3 Example Image Prompts Test SUCCESSFUL!")

def test_video_model():
    print("\n=== Running SAM3 Video Feature Test ===")
    
    # Monkey-patch the submodule bug where Sam3MultiplexTrackingWithInteractivity.init_state
    # does not accept the 'offload_state_to_cpu' argument passed by sam3_base_predictor.py
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
        print("Successfully patched Sam3MultiplexTrackingWithInteractivity.init_state")
    except ImportError as e:
        print(f"Could not patch Sam3MultiplexTrackingWithInteractivity: {e}")

    from sam3.model_builder import build_sam3_predictor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM3 video predictor on {device}...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    bpe_path = os.path.join(parent_dir, "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_predictor(bpe_path=bpe_path, version="sam3.1", compile=False, async_loading_frames=False, use_fa3=False)
    
    frames_dir = "test_video_frames"
    create_synthetic_frames(frames_dir)
    
    try:
        # Start session
        print("Initializing session...")
        response = model.handle_request({"type": "start_session", "resource_path": frames_dir})
        session_id = response["session_id"]
        print(f"Session started with ID: {session_id}")
        
        # Add prompt "circle" on frame 0
        prompt = "circle"
        print(f"Adding text prompt: '{prompt}' on frame 0...")
        model.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt,
        })
        
        # Propagate video tracking
        print("Propagating tracking...")
        propagated_frames = 0
        for res in model.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
            frame_idx = res.get("frame_index")
            if frame_idx is not None:
                outputs = res.get("outputs", {})
                out_obj_ids = outputs.get("out_obj_ids", [])
                print(f"  Frame {frame_idx}: tracked object IDs: {out_obj_ids}")
                propagated_frames += 1
                
        print(f"Propagation completed for {propagated_frames} frames.")
        assert propagated_frames > 0, "No frames were tracked!"
        print("SAM3 Video Model Test SUCCESSFUL!")
    finally:
        # Cleanup temporary frames directory
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)

if __name__ == "__main__":
    load_env_vars()
    # Autocast to BF16 for faster and memory-efficient execution
    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
        test_image_model()
        test_example_image_prompts()
        test_video_model()
EOF

# Run unit tests
uv run python "$TEMP_TEST_FILE"

echo ""
echo "========================================="
echo "Running SAM3 Asset Processing Tests"
echo "========================================="

# Run asset processing for both images and videos
uv run python process_assets.py --mode all

echo ""
echo "========================================="
echo "All E2E tests completed successfully!"
echo "========================================="
exit 0
