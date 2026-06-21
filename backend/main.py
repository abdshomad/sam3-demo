import sys
import timm.layers
sys.modules["timm.models.layers"] = timm.layers
import os
import uuid
import torch
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add workspace and sam3 to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
sam3_root = os.path.join(workspace_dir, "sam3")
if sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)

# Set environment cache variable
os.environ["HF_HOME"] = os.path.join(workspace_dir, ".hf_cache")
env_path = os.path.join(workspace_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                break

# Configure matplotlib for headless running
import matplotlib
matplotlib.use('Agg')

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
except Exception:
    pass

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
except Exception:
    pass

from sam3.model_builder import build_sam3_image_model, build_sam3_predictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.geometry_encoders import Prompt
from process_assets import save_mask_overlay

app = FastAPI(title="SAM3 Dashboard API", docs_url="/api/docs", openapi_url="/api/openapi.json")

# CORS middleware for Next.js routing (even though Nginx proxies, CORS is good practice)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Model Managers for lazy loading
class ModelManager:
    def __init__(self):
        self.image_model = None
        self.image_processor = None
        self.video_predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bpe_path = os.path.join(workspace_dir, "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
        
    def get_image_processor(self):
        if self.image_processor is None:
            print("Lazy loading image model...")
            self.image_model = build_sam3_image_model(bpe_path=self.bpe_path, device=self.device, load_from_HF=True, enable_inst_interactivity=True)
            self.image_processor = Sam3Processor(self.image_model)
        return self.image_processor
        
    def get_video_predictor(self):
        current_version = os.getenv("SAM_VERSION", "sam3.1")
        # Map sam2 to sam3 since build_sam3_predictor takes "sam3" or "sam3.1"
        builder_version = "sam3" if current_version == "sam2" else current_version
        
        # If version changed or predictor not loaded yet
        if self.video_predictor is None or getattr(self, "_loaded_sam_version", None) != current_version:
            print(f"Lazy loading video model with version {current_version} (builder version: {builder_version})...")
            # Free old predictor memory if it exists
            if self.video_predictor is not None:
                del self.video_predictor
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            self.video_predictor = build_sam3_predictor(
                bpe_path=self.bpe_path, 
                version=builder_version, 
                compile=False, 
                async_loading_frames=False, 
                use_fa3=False
            )
            self.video_predictor.model.batched_grounding_batch_size = 2
            self.video_predictor.model.postprocess_batch_size = 2
            self._loaded_sam_version = current_version
            
        return self.video_predictor

model_manager = ModelManager()

# Ensure directories exist
os.makedirs(os.path.join(workspace_dir, "results/interactive"), exist_ok=True)
app.mount("/api/original", StaticFiles(directory=os.path.join(workspace_dir, "sam3/assets")), name="original")
app.mount("/api/results", StaticFiles(directory=os.path.join(workspace_dir, "results")), name="results")

# Input models
class ClickPoint(BaseModel):
    x: float
    y: float
    is_positive: bool

class InferenceRequest(BaseModel):
    asset_type: str  # "image" or "video"
    asset_path: str  # e.g., "images/truck.jpg", "videos/bedroom.mp4"
    prompt: str

class ClickInferenceRequest(BaseModel):
    asset_path: str  # e.g., "images/truck.jpg"
    clicks: List[ClickPoint]
    prompt: Optional[str] = None
    session_id: Optional[str] = None

# Interactive point helper
def add_point_prompt(processor, x, y, label_bool, state):
    if "backbone_out" not in state:
        raise ValueError("You must call set_image before prompting")
        
    if "language_features" not in state["backbone_out"]:
        dummy_text_outputs = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        state["backbone_out"].update(dummy_text_outputs)
        
    if "geometric_prompt" not in state or state["geometric_prompt"] is None or state["geometric_prompt"].point_embeddings is None:
        device = processor.device
        state["geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device=device),
            box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            point_embeddings=torch.zeros(0, 1, 2, device=device),
            point_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device=device, dtype=torch.long),
        )
        
    # Scale points from normalized range [0, 1] to target processor resolution
    resolution = getattr(processor, "resolution", 1008)
    scaled_x = x * resolution
    scaled_y = y * resolution
    
    # Append points: shape [1, 1, 2] for points, [1, 1] for labels
    points = torch.tensor([[[scaled_x, scaled_y]]], device=processor.device, dtype=torch.float32).view(1, 1, 2)
    labels = torch.tensor([[label_bool]], device=processor.device, dtype=torch.long).view(1, 1)
    state["geometric_prompt"].append_points(points, labels)
    
    return processor._forward_grounding(state)

# Sessions storage to preserve states between click calls
interactive_sessions = {}

def transcode_to_h264(temp_path: str, final_path: str):
    import subprocess
    import shutil
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            final_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"ffmpeg transcoding failed: {e}. Falling back to raw copy.")
        if os.path.exists(temp_path):
            shutil.move(temp_path, final_path)

@app.get("/api/health")
def health_check():
    return {"status": "ok", "gpu_available": torch.cuda.is_available()}

# Configuration Models
class ConfigResponse(BaseModel):
    vl_model: str
    parse_model: str
    sam_version: str
    available_vl_models: List[Dict[str, str]]
    available_parse_models: List[Dict[str, str]]
    available_sam_versions: List[Dict[str, str]]

class UpdateConfigRequest(BaseModel):
    vl_model: str
    parse_model: str
    sam_version: str

@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    # Read model configurations from environment/env
    vl_model = os.getenv("VL_MODEL", "Qwen/Qwen3-VL-8B-Thinking")
    parse_model = os.getenv("PARSE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    sam_version = os.getenv("SAM_VERSION", "sam3.1")
    
    available_vl_models = [
        {"id": "Qwen/Qwen3-VL-8B-Thinking", "name": "Qwen3-VL-8B-Thinking (Default, Thinking)", "description": "Flagship multimodal thinking model (8B)."},
        {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "name": "Qwen2.5-VL-7B-Instruct (Standard)", "description": "Standard fast multimodal model (7B)."},
        {"id": "Qwen/Qwen2.5-VL-3B-Instruct", "name": "Qwen2.5-VL-3B-Instruct (Lightweight)", "description": "Fast and light multimodal model (3B)."}
    ]
    
    available_parse_models = [
        {"id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen2.5-0.5B-Instruct (Ultralight, Cached)", "description": "Ultralight model for extremely fast inference and low memory footprint."},
        {"id": "Qwen/Qwen3.6-35B-A3B-FP8", "name": "Qwen3.6-35B-A3B-FP8 (Default MoE)", "description": "Highly efficient 35B Mixture-of-Experts (3B active parameters) text model."},
        {"id": "Qwen/Qwen3.6-27B-FP8", "name": "Qwen3.6-27B-FP8 (Dense)", "description": "Flagship 27B dense text model."},
        {"id": "Qwen/Qwen3-8B-FP8", "name": "Qwen3-8B-FP8 (Lightweight)", "description": "Fast and lightweight 8B text model."}
    ]

    available_sam_versions = [
        {"id": "sam3.1", "name": "SAM 3.1 (Default, Object Multiplex)", "description": "Latest release with Object Multiplex for joint multi-object tracking and compilation support."},
        {"id": "sam3", "name": "SAM 3 (Base)", "description": "Standard base SAM3 model for video tracking and segmenting."},
        {"id": "sam2", "name": "SAM 2 (SAM 3 backward-compatible mode)", "description": "SAM 3 base running in backward-compatibility mode for SAM 2 tasks."}
    ]
    
    return {
        "vl_model": vl_model,
        "parse_model": parse_model,
        "sam_version": sam_version,
        "available_vl_models": available_vl_models,
        "available_parse_models": available_parse_models,
        "available_sam_versions": available_sam_versions
    }

@app.post("/api/config/update-models")
def update_models(req: UpdateConfigRequest, background_tasks: BackgroundTasks):
    # Rewrite the .env file with updated models
    vl_model = req.vl_model.strip()
    parse_model = req.parse_model.strip()
    sam_version = req.sam_version.strip()
    
    # Read current .env content
    env_file = os.path.join(workspace_dir, ".env")
    current_env = {}
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith("#") and "=" in line_stripped:
                    key, val = line_stripped.split("=", 1)
                    current_env[key.strip()] = val.strip()
                    
    # Update values
    current_env["VL_MODEL"] = vl_model
    current_env["PARSE_MODEL"] = parse_model
    current_env["SAM_VERSION"] = sam_version
    
    # Write back
    with open(env_file, "w") as f:
        for k, v in current_env.items():
            f.write(f"{k}={v}\n")
            
    # Check if Qwen models changed
    old_vl_model = os.environ.get("VL_MODEL", "").strip()
    old_parse_model = os.environ.get("PARSE_MODEL", "").strip()
    qwen_models_changed = (vl_model != old_vl_model or parse_model != old_parse_model)

    # Set environment variables for the current backend process as well
    os.environ["VL_MODEL"] = vl_model
    os.environ["PARSE_MODEL"] = parse_model
    os.environ["SAM_VERSION"] = sam_version

    if qwen_models_changed:
        # Background task to restart the containers
        def restart_containers():
            import subprocess
            try:
                print(f"Triggering background docker compose restart for models: VL={vl_model}, PARSE={parse_model}")
                project_name = os.getenv("COMPOSE_PROJECT_NAME", "sam3-demo-jun-2026-v2")
                res = subprocess.run(
                    ["docker", "compose", "--project-name", project_name, "up", "-d", "--force-recreate", "qwen3-vl", "qwen3-6"],
                    cwd=workspace_dir,
                    capture_output=True,
                    text=True
                )
                print(f"Docker compose restart stdout: {res.stdout}")
                if res.stderr:
                    print(f"Docker compose restart stderr: {res.stderr}")
            except Exception as e:
                print(f"Failed to restart docker compose containers: {e}")

        background_tasks.add_task(restart_containers)
        return {"success": True, "message": f"Updated configurations (SAM version set to {sam_version}). Recreating Qwen container services in background..."}
    else:
        return {"success": True, "message": f"Updated configurations (SAM version set to {sam_version}). No container restart needed since Qwen models did not change."}

def get_images_used_in_examples():
    # Only include groceries, test image and truck
    return ["images/groceries.jpg", "images/test_image.jpg", "images/truck.jpg"]


@app.get("/api/assets")
def get_assets():
    # Scan original assets
    original_images = get_images_used_in_examples()
    original_videos = ["videos/bedroom.mp4"]

    # Scan static results
    results_dir = os.path.join(workspace_dir, "results")
    result_images = []
    result_videos = []
    
    res_img_dir = os.path.join(results_dir, "images")
    if os.path.exists(res_img_dir):
        result_images = [os.path.join("images", f) for f in os.listdir(res_img_dir) if f.lower().endswith(('.png', '.jpg'))]
        
    res_vid_dir = os.path.join(results_dir, "videos")
    if os.path.exists(res_vid_dir):
        result_videos = [os.path.join("videos", f) for f in os.listdir(res_vid_dir) if f.lower().endswith('.mp4')]
        
    return {
        "originals": {
            "images": sorted(list(set(original_images))),
            "videos": sorted(list(set(original_videos)))
        },
        "results": {
            "images": sorted(list(set(result_images))),
            "videos": sorted(list(set(result_videos)))
        }
    }

def get_gif_frames_dir(gif_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(gif_path))[0]
    target_dir = os.path.join(workspace_dir, "results/interactive", f"gif_cache_{base_name}")
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        return target_dir
        
    os.makedirs(target_dir, exist_ok=True)
    gif = Image.open(gif_path)
    idx = 0
    while True:
        try:
            gif.seek(idx)
            frame = gif.convert("RGB")
            frame.save(os.path.join(target_dir, f"{idx:05d}.jpg"), "JPEG")
            idx += 1
        except EOFError:
            break
    print(f"Extracted {idx} frames from GIF {gif_path} to {target_dir}")
    return target_dir

@app.post("/api/inference")
async def run_inference(req: InferenceRequest):
    session_id = str(uuid.uuid4())
    device = model_manager.device
    
    input_path = os.path.join(workspace_dir, "sam3/assets", req.asset_path)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Asset not found: {req.asset_path}")
        
    if req.asset_type == "image":
        try:
            processor = model_manager.get_image_processor()
            image = Image.open(input_path)
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                state = processor.set_image(image)
                output = processor.set_text_prompt(state=state, prompt=req.prompt)
                
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            output_filename = f"{session_id}.png"
            output_path = os.path.join(workspace_dir, "results/interactive", output_filename)
            
            save_mask_overlay(input_path, masks, boxes, scores, output_path, req.prompt)
            
            # Crop and classify the segmented objects
            crops = await classify_and_crop_segmented_objects(input_path, masks, boxes, session_id)
            
            return {
                "success": True,
                "output_url": f"/api/results/interactive/{output_filename}",
                "session_id": session_id,
                "crops": crops
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image inference failed: {str(e)}")
            
    elif req.asset_type == "video":
        try:
            predictor = model_manager.get_video_predictor()
            
            # If GIF, extract frames and treat as image directory
            is_gif = req.asset_path.lower().endswith(".gif")
            tracking_path = input_path
            if is_gif:
                tracking_path = get_gif_frames_dir(input_path)
                
            print(f"Starting session for video: {tracking_path}")
            response = predictor.handle_request({
                "type": "start_session", 
                "resource_path": tracking_path,
                "offload_video_to_cpu": True
            })
            vid_session_id = response["session_id"]
            
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": vid_session_id,
                "frame_index": 0,
                "text": req.prompt,
            })
            
            # Setup video writing
            if tracking_path.endswith(".mp4"):
                cap = cv2.VideoCapture(tracking_path)
                frames_rgb = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
                h, w, _ = frames_rgb[0].shape
            else:
                # Frame folder: 0001 or extracted gif frames
                first_frame_path = os.path.join(tracking_path, "00000.jpg")
                if not os.path.exists(first_frame_path):
                    first_frame_path = os.path.join(tracking_path, "0.jpg")
                first_frame = Image.open(first_frame_path)
                w, h = first_frame.size
                
            output_filename = f"{session_id}.mp4"
            temp_output_filename = f"raw_{session_id}.mp4"
            output_path = os.path.join(workspace_dir, "results/interactive", output_filename)
            temp_output_path = os.path.join(workspace_dir, "results/interactive", temp_output_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_output_path, fourcc, 10.0, (w, h))
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                stream = predictor.handle_stream_request({
                    "type": "propagate_in_video", 
                    "session_id": vid_session_id, 
                    "max_frame_num_to_track": 30
                })
                for res in stream:
                    frame_idx = res.get("frame_index")
                    if frame_idx is not None:
                        outputs = res.get("outputs", {})
                        out_obj_ids = outputs.get("out_obj_ids", [])
                        binary_masks = outputs.get("out_binary_masks")
                        
                        if binary_masks is not None:
                            if isinstance(binary_masks, torch.Tensor):
                                binary_masks = binary_masks.cpu().numpy()
                                
                            if tracking_path.endswith(".mp4"):
                                frame_input = frames_rgb[frame_idx]
                            else:
                                frame_input = os.path.join(tracking_path, f"{frame_idx}.jpg")
                                if not os.path.exists(frame_input):
                                    frame_input = os.path.join(tracking_path, f"{frame_idx:05d}.jpg")
                                    
                            overlay_rgb = save_mask_overlay(
                                frame_input,
                                binary_masks,
                                None,
                                None,
                                None,
                                f"obj_{out_obj_ids[0] if len(out_obj_ids) > 0 else 0}"
                            )
                            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                            video_writer.write(overlay_bgr)
                            
            video_writer.release()
            transcode_to_h264(temp_output_path, output_path)
            predictor.handle_request({"type": "close_session", "session_id": vid_session_id})
            
            return {
                "success": True,
                "output_url": f"/api/results/interactive/{output_filename}",
                "session_id": session_id,
                "crops": []
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Video tracking failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Invalid asset type")

@app.post("/api/interactive-click")
async def run_interactive_click(req: ClickInferenceRequest):
    session_id = req.session_id or str(uuid.uuid4())
    device = model_manager.device
    
    input_path = os.path.join(workspace_dir, "sam3/assets", req.asset_path)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Asset not found: {req.asset_path}")
        
    try:
        processor = model_manager.get_image_processor()
        image = Image.open(input_path)
        
        # Check if we have cached state for this session
        if session_id in interactive_sessions:
            state = interactive_sessions[session_id]
        else:
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                state = processor.set_image(image)
            interactive_sessions[session_id] = state
            
        # Reset prompts before adding new ones
        processor.reset_all_prompts(state)
        
        # If there's a text prompt, apply it first
        if req.prompt:
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                processor.set_text_prompt(state=state, prompt=req.prompt)
                
        # Now add all clicks
        if len(req.clicks) > 0:
            orig_w = state["original_width"]
            orig_h = state["original_height"]
            
            # Format coordinates for predict_inst
            point_coords = np.array([[click.x * orig_w, click.y * orig_h] for click in req.clicks], dtype=np.float32)
            point_labels = np.array([1 if click.is_positive else 0 for click in req.clicks], dtype=np.int32)
            
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                masks, scores, logits = model_manager.image_model.predict_inst(
                    state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                
            best_idx = np.argmax(scores)
            masks_tensor = torch.from_numpy(masks[best_idx]).unsqueeze(0).unsqueeze(0).to(device)
            scores_tensor = torch.tensor([scores[best_idx]], device=device)
            
            y_indices, x_indices = np.where(masks[best_idx])
            if len(y_indices) > 0:
                y_min, y_max = y_indices.min(), y_indices.max()
                x_min, x_max = x_indices.min(), x_indices.max()
                boxes_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32, device=device)
            else:
                boxes_tensor = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        else:
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                output = processor._forward_grounding(state)
            masks_tensor = output["masks"]
            boxes_tensor = output["boxes"]
            scores_tensor = output["scores"]
            
        output_filename = f"click_{session_id}.png"
        output_path = os.path.join(workspace_dir, "results/interactive", output_filename)
        
        save_mask_overlay(input_path, masks_tensor, boxes_tensor, scores_tensor, output_path, req.prompt or "object")
        
        crops = await classify_and_crop_segmented_objects(input_path, masks_tensor, boxes_tensor, session_id)
        
        return {
            "success": True,
            "output_url": f"/api/results/interactive/{output_filename}",
            "session_id": session_id,
            "crops": crops
        }
    except Exception as e:
        # Clear caching if it failed
        if session_id in interactive_sessions:
            del interactive_sessions[session_id]
        raise HTTPException(status_code=500, detail=f"Click inference failed: {str(e)}")

class SubObjectParser(BaseModel):
    name: str
    attributes: List[str] = []

class ObjectParser(BaseModel):
    name: str
    attributes: List[str] = []
    sub_objects: List[SubObjectParser] = []

class BreakdownParser(BaseModel):
    objects: List[ObjectParser]

class ClassificationCandidate(BaseModel):
    class_name: str
    confidence: float

class ClassificationBreakdown(BaseModel):
    candidates: List[ClassificationCandidate]

class AssetDescribeBreakdown(BaseModel):
    objects: List[ObjectParser]
    candidates: List[ClassificationCandidate] = []

class SimpleObjectExtractor(BaseModel):
    objects: List[str]

class DescribeRequest(BaseModel):
    asset_path: str  # e.g., "images/truck.jpg" or "videos/bedroom.mp4"
    prompt: Optional[str] = "Describe this asset in detail."
    want_breakdown: Optional[bool] = False

def parse_pydantic_breakdown(text: str, fallback_text: Optional[str] = None) -> list[dict]:
    import re
    import json
    
    # Clean thinking tags if present
    cleaned_text = text
    if "</think>" in text:
        cleaned_text = text.split("</think>", 1)[1]
    elif "</thinking>" in text:
        cleaned_text = text.split("</thinking>", 1)[1]
    else:
        cleaned_text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
        cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
        
    json_block = ""
    match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
    if match:
        json_block = match.group(0)
        
    objects_data = []
    
    if json_block:
        try:
            # Parse and validate using Pydantic models
            parsed_data = BreakdownParser.model_validate_json(json_block)
            
            for obj in parsed_data.objects:
                name = obj.name.strip()
                attrs = [a.strip() for a in obj.attributes if a.strip()]
                if not name:
                    continue
                    
                attr_str = " ".join(attrs)
                main_prompt = f"{attr_str} {name}".strip()
                
                sub_objects_list = []
                for sub in obj.sub_objects:
                    sub_name = sub.name.strip()
                    sub_attrs = [sa.strip() for sa in sub.attributes if sa.strip()]
                    if not sub_name:
                        continue
                        
                    sub_attr_str = " ".join(sub_attrs)
                    sub_prompt = f"{sub_attr_str} {sub_name} of {main_prompt}".strip()
                    
                    sub_objects_list.append({
                        "name": sub_name,
                        "prompt": sub_prompt,
                        "attributes": sub_attrs
                    })
                    
                objects_data.append({
                    "name": name,
                    "prompt": main_prompt,
                    "attributes": attrs,
                    "sub_objects": sub_objects_list
                })
        except Exception as e:
            print(f"Pydantic parsing failed: {e}")
            
    # Fallback to bulleted lists
    if not objects_data:
        fallback_items = []
        text_to_parse = fallback_text if fallback_text is not None else text
        for line in text_to_parse.splitlines():
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                item = line[2:].strip().strip('.,;":\'')
                if item:
                    fallback_items.append(item)
                    
        seen = set()
        deduped = []
        for item in fallback_items:
            if item.lower() not in seen:
                seen.add(item.lower())
                deduped.append(item)
                
        for item in deduped:
            objects_data.append({
                "name": item,
                "prompt": item,
                "attributes": [],
                "sub_objects": []
            })
            
    return objects_data

async def get_active_model(base_url: str, default_model: str) -> str:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                if data and "data" in data and len(data["data"]) > 0:
                    model_id = data["data"][0]["id"]
                    print(f"Retrieved active model {model_id} from {base_url}")
                    return model_id
    except Exception as e:
        print(f"Failed to fetch active model from {base_url}: {e}. Using fallback {default_model}")
    return default_model

async def classify_crop(marked_filename: str) -> List[dict]:
    import httpx
    import re
    import json
    
    container_file_path = f"/app/results/interactive/{marked_filename}"
    media_payload = {"image_url": {"url": f"file://{container_file_path}"}}
    
    vl_prompt = (
        "Identify the object enclosed within the red bounding box in this image. "
        "Describe the object and list multiple likely category names/ontology terms for it, "
        "along with your estimated confidence level (0.0 to 1.0) for each candidate. "
        "Explain your reasoning for the categories."
    )
    
    active_vl_model = await get_active_model("http://qwen3-vl:8000", os.getenv("VL_MODEL", "Qwen/Qwen3-VL-8B-Thinking"))
    vl_payload = {
        "model": active_vl_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vl_prompt},
                    {"type": "image_url", **media_payload}
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"
    }
    
    vl_description = ""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post("http://qwen3-vl:8000/v1/chat/completions", json=vl_payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                vl_description = result["choices"][0]["message"]["content"].strip()
                if "</think>" in vl_description:
                    vl_description = vl_description.split("</think>")[-1].strip()
                elif "</thinking>" in vl_description:
                    vl_description = vl_description.split("</thinking>")[-1].strip()
            else:
                print(f"Qwen-VL request failed in classify_crop: {response.text}")
    except Exception as e:
        print(f"Qwen-VL query failed in classify_crop: {e}")
        
    if not vl_description:
        return {"description": "No description available.", "candidates": [{"class_name": "unknown object", "confidence": 1.0}]}
        
    parser_prompt = (
        "Extract all likely candidate object classes and their confidence scores from the text description below.\n\n"
        "Text:\n"
        f"{vl_description}\n\n"
        "Format your response strictly as a valid JSON object matching this schema (do NOT output any other text or reasoning):\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "class_name": "[concise category name, e.g. pickup truck]",\n'
        '      "confidence": [float between 0.0 and 1.0]\n'
        "    }\n"
        "  ]\n"
        "}"
    )
    
    active_parse_model = await get_active_model("http://qwen3-6:8000", os.getenv("PARSE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser_payload = {
        "model": active_parse_model,
        "messages": [
            {
                "role": "user",
                "content": parser_prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.1
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("http://qwen3-6:8000/v1/chat/completions", json=parser_payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()
                    
                json_block = ""
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    json_block = match.group(0)
                    
                if json_block:
                    try:
                        # Clean single-line and multi-line comments
                        lines = []
                        for line in json_block.splitlines():
                            stripped = line.strip()
                            if stripped.startswith("//") or stripped.startswith("/*"):
                                continue
                            if "//" in line:
                                parts = line.split("//", 1)
                                if parts[0].count('"') % 2 == 0:
                                    line = parts[0]
                            lines.append(line)
                        cleaned_json_block = "\n".join(lines)
                        
                        # Clean trailing commas
                        cleaned_json_block = re.sub(r',\s*\]', ']', cleaned_json_block)
                        cleaned_json_block = re.sub(r',\s*\}', '}', cleaned_json_block)
                        
                        parsed_data = ClassificationBreakdown.model_validate_json(cleaned_json_block)
                        sorted_candidates = sorted(parsed_data.candidates, key=lambda x: x.confidence, reverse=True)
                        return {
                            "description": vl_description,
                            "candidates": [{"class_name": c.class_name.strip("`'\" \n\t."), "confidence": c.confidence} for c in sorted_candidates]
                        }
                    except Exception as parse_err:
                        print(f"Pydantic parsing failed in classify_crop: {parse_err}. Trying manual extraction...", flush=True)
                        candidates_list = []
                        try:
                            candidate_matches = re.findall(r'"class_name":\s*"([^"]+)"\s*,\s*"confidence":\s*([0-9.]+)', json_block)
                            if not candidate_matches:
                                candidate_matches = re.findall(r'"confidence":\s*([0-9.]+)\s*,\s*"class_name":\s*"([^"]+)"', json_block)
                                candidate_matches = [(name, float(conf)) for conf, name in candidate_matches]
                            else:
                                candidate_matches = [(name, float(conf)) for name, conf in candidate_matches]
                            
                            for name, conf in candidate_matches:
                                candidates_list.append({
                                    "class_name": name.strip("`'\" \n\t."),
                                    "confidence": conf
                                })
                            candidates_list = sorted(candidates_list, key=lambda x: x["confidence"], reverse=True)
                        except Exception as fallback_err:
                            print(f"Manual fallback extraction in classify_crop also failed: {fallback_err}", flush=True)
                        
                        if candidates_list:
                            return {
                                "description": vl_description,
                                "candidates": candidates_list
                            }
    except Exception as e:
        print(f"Parser model failed to structure candidates list: {e}")
        
    return {"description": vl_description or "Failed to parse description.", "candidates": [{"class_name": "unknown object", "confidence": 1.0}]}

async def classify_and_crop_segmented_objects(image_path_or_array, masks, boxes, session_id: str):
    from PIL import ImageDraw
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().float().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().float().numpy()

    if masks is None or len(masks) == 0:
        return []

    # Load original image
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert("RGBA")
    else:
        img = Image.fromarray(image_path_or_array).convert("RGBA")

    w, h = img.size
    crops_list = []

    for idx, mask in enumerate(masks):
        if mask.ndim == 3:
            mask = mask[0]
        mask_bool = mask.astype(bool)
        
        # Resize mask if it does not match the image dimensions
        if mask_bool.shape != (h, w):
            mask_pil = Image.fromarray((mask_bool * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
            mask_bool_resized = np.array(mask_pil) > 127
        else:
            mask_bool_resized = mask_bool

        # Make the background white where mask is false
        img_np = np.array(img)
        img_np[~mask_bool_resized, 3] = 0
        masked_img = Image.fromarray(img_np)

        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        cropped_full = Image.alpha_composite(bg, masked_img).convert("RGB")

        # Bounding box coordinates
        x1, y1, x2, y2 = 0, 0, w, h
        if boxes is not None and idx < len(boxes):
            box = boxes[idx]
            x1, y1, x2, y2 = map(int, box)
        else:
            # compute from mask
            y_indices, x_indices = np.where(mask_bool_resized)
            if len(x_indices) > 0 and len(y_indices) > 0:
                x1, x2 = x_indices.min(), x_indices.max()
                y1, y2 = y_indices.min(), y_indices.max()

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 > x1 and y2 > y1:
            crop_img = cropped_full.crop((x1, y1, x2, y2))
        else:
            crop_img = cropped_full

        # Save crop image under results directory
        crop_filename = f"crop_{session_id}_{idx}.png"
        crop_path = os.path.join(workspace_dir, "results/interactive", crop_filename)
        crop_img.save(crop_path, "PNG")

        # Create a marked full image (original full image with a red bounding box) to give context to Qwen-VL
        marked_img = img.convert("RGB")
        draw_marked = ImageDraw.Draw(marked_img)
        draw_marked.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=5)
        
        # Save marked full image
        marked_filename = f"marked_{session_id}_{idx}.png"
        marked_path = os.path.join(workspace_dir, "results/interactive", marked_filename)
        marked_img.save(marked_path, "PNG")

        # Classify using Qwen VL + Qwen 3.6
        classification_result = await classify_crop(marked_filename)
        description = classification_result["description"]
        candidates = classification_result["candidates"]
        class_name = candidates[0]["class_name"] if len(candidates) > 0 else "unknown object"

        crops_list.append({
            "crop_url": f"/api/results/interactive/{crop_filename}",
            "class_name": class_name,
            "description": description,
            "candidates": candidates
        })

    return crops_list

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy en_core_web_sm model...", flush=True)
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def extract_objects_with_spacy(text: str) -> list[str]:
    nlp = get_nlp()
    doc = nlp(text)
    
    # Comprehensive blacklist of abstract and meta words
    blacklist = {
        "it", "they", "them", "he", "she", "we", "you", "i", "us", "him", "her", "me",
        "myself", "himself", "herself", "itself", "themselves", "ourselves", "yourselves",
        "this", "that", "these", "those", "something", "anything", "nothing", "everything",
        "someone", "anyone", "everyone", "no one", "somebody", "anybody", "everybody", "nobody",
        "one", "ones", "others", "another", "both", "all", "any", "some", "none", "few",
        "many", "several", "more", "most", "other", "such", "own", "same",
        
        # Visual / Layout
        "image", "photo", "picture", "video", "asset", "frame", "pixels", "area", "scene",
        "view", "shot", "angle", "perspective", "background", "foreground", "midground",
        "middleground", "center", "centre", "left", "right", "top", "bottom", "side", "sides",
        "corner", "corners", "border", "edge", "edges", "boundary", "boundaries",
        
        # Structural / Text / Meta
        "step", "steps", "list", "item", "items", "entry", "entries", "row", "column", "table",
        "section", "part", "parts", "component", "components", "attribute", "attributes",
        "feature", "features", "detail", "details", "description", "descriptions", "text",
        "prompt", "output", "object", "objects", "subject", "subjects", "thing", "things",
        "element", "elements", "entity", "entities", "target", "targets", "focus", "focal",
        
        # Quality / Class / Number / Math
        "color", "colour", "colors", "colours", "shade", "hue", "value", "tint", "tone",
        "shape", "shapes", "form", "forms", "size", "sizes", "dimension", "dimensions",
        "width", "height", "depth", "length", "scale", "position", "positions", "location",
        "locations", "place", "places", "space", "spaces", "distance", "distances", "range",
        "ranges", "texture", "textures", "material", "materials", "pattern", "patterns",
        "design", "designs", "style", "styles", "aspect", "aspects", "instance", "instances",
        "segment", "segments", "crop", "crops", "mask", "masks", "bounding", "box", "boxes",
        "presence", "absence", "visible", "visibility", "viewable", "appearance",
        "main", "secondary", "primary", "tertiary", "major", "minor", "key", "sole", "only",
        "single", "double", "triple", "type", "types", "kind", "kinds", "sort", "sorts",
        "class", "classes", "category", "categories", "analysis", "detection", "segmentation",
        "classification", "candidate", "candidates", "label", "labels", "name", "names",
        "value", "values", "addition", "subtraction", "multiplication", "division", "percent",
        "percentage", "ratio", "ratios", "number", "numbers", "digit", "digits", "amount",
        "amounts", "quantity", "quantities", "example", "examples", "specimen", "specimens",
        "sample", "samples", "way", "manner", "fashion", "method", "methods", "mode", "modes",
        "time", "times", "date", "dates", "year", "years", "month", "months", "day", "days",
        "hour", "hours", "minute", "minutes", "second", "seconds", "version", "versions",
        "source", "sources", "destination", "destinations", "level", "levels", "degree",
        "degrees", "extent", "extents", "stage", "stages", "contrast", "brightness", "saturation",
        "exposure", "lighting", "light", "shadow", "shadows", "reflection", "reflections", "glare",
        "etc", "etcetera"
    }

    def clean_chunk_tokens(chunk) -> str:
        skip_pos = {"ADV", "DET", "PRON", "ADP", "SCONJ", "CCONJ", "PUNCT", "PART"}
        start_idx = 0
        while start_idx < len(chunk) and chunk[start_idx].pos_ in skip_pos:
            start_idx += 1
        
        end_idx = len(chunk)
        while end_idx > start_idx and chunk[end_idx - 1].pos_ in {"PUNCT", "PART", "CCONJ", "SCONJ", "ADP"}:
            end_idx -= 1
            
        if start_idx >= end_idx:
            return ""
        return chunk[start_idx:end_idx].text.strip()

    extracted = []
    
    # 1. Extract noun chunks
    for chunk in doc.noun_chunks:
        if chunk.root.is_stop or chunk.root.pos_ == "PRON" or chunk.root.like_num:
            continue
        if chunk.root.lemma_.lower() in blacklist or chunk.root.text.lower() in blacklist:
            continue
            
        chunk_text = clean_chunk_tokens(chunk)
        chunk_text = chunk_text.strip(".,;:!?`'\" \n\t")
        if not chunk_text or len(chunk_text) <= 1:
            continue
            
        lower_chunk = chunk_text.lower()
        if lower_chunk in blacklist:
            continue
        if any(lower_chunk.endswith(" " + b) for b in blacklist):
            continue
            
        extracted.append(chunk_text)
        
    # 2. Extract individual nouns/propn only if not part of already extracted phrases
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            if token.is_stop or token.pos_ == "PRON" or token.like_num:
                continue
            if token.lemma_.lower() in blacklist or token.text.lower() in blacklist:
                continue
                
            noun_text = token.text.strip(".,;:!?`'\" \n\t")
            if noun_text and len(noun_text) > 1:
                noun_lower = noun_text.lower()
                if noun_lower not in blacklist:
                    # Check if it's already part of some extracted chunk
                    is_subpart = False
                    for phrase in extracted:
                        if noun_lower in [w.lower() for w in phrase.split()]:
                            is_subpart = True
                            break
                    if not is_subpart:
                        extracted.append(noun_text)
                    
    # 3. Extract Named Entities
    for ent in doc.ents:
        if ent.label_ in ("PRODUCT", "ORG", "PERSON", "FAC", "GPE", "LOC"):
            ent_text = ent.text.strip(".,;:!?`'\" \n\t")
            if ent_text and len(ent_text) > 1:
                ent_clean = clean_chunk_tokens(ent)
                ent_lower = ent_clean.lower()
                if ent_lower not in blacklist and not any(ent_lower.endswith(" " + b) for b in blacklist):
                    # Check if already subpart
                    is_subpart = False
                    for phrase in extracted:
                        if ent_lower in [w.lower() for w in phrase.split()]:
                            is_subpart = True
                            break
                    if not is_subpart:
                        extracted.append(ent_clean)

    # Clean and deduplicate (case-insensitive deduplication, but preserving original case)
    seen = set()
    unique_objects = []
    for item in extracted:
        item_lower = item.lower()
        if item_lower not in seen:
            seen.add(item_lower)
            unique_objects.append(item)
            
    # Sort by length descending
    unique_objects.sort(key=lambda x: len(x), reverse=True)
    return unique_objects

@app.post("/api/describe")
async def describe_asset(req: DescribeRequest):
    import httpx
    import re
    
    input_path = os.path.join(workspace_dir, "sam3/assets", req.asset_path)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Asset not found: {req.asset_path}")

    container_file_path = f"/app/sam3/assets/{req.asset_path}"
    is_video = req.asset_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.gif'))
    
    if is_video:
        media_type = "video_url"
        media_payload = {"video_url": {"url": f"file://{container_file_path}"}}
    else:
        media_type = "image_url"
        media_payload = {"image_url": {"url": f"file://{container_file_path}"}}

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"
    }

    if req.want_breakdown:
        # Step 1: Query Qwen-VL to get a detailed visual description
        vl_prompt = (
            "Identify and describe ALL possible objects present in this image/video, including the main object and all secondary or background objects. "
            "For each object, output the object name, its key parts/components, and the attributes (like color, shape, relative position, or texture) of the object and its parts. "
            "Be extremely detailed and comprehensive. Do NOT focus only on the main object; list and describe every single object visible in the asset along with its parts and attributes."
        )
        active_vl_model = await get_active_model("http://qwen3-vl:8000", os.getenv("VL_MODEL", "Qwen/Qwen3-VL-8B-Thinking"))
        vl_payload = {
            "model": active_vl_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vl_prompt},
                        {"type": media_type, **media_payload}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post("http://qwen3-vl:8000/v1/chat/completions", json=vl_payload, headers=headers)
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"vLLM Qwen-VL server error: {response.text}")
                
                result = response.json()
                vl_description = result["choices"][0]["message"]["content"]
                
                # Clean thinking tokens if they leaked into description
                if "</think>" in vl_description:
                    vl_description = vl_description.split("</think>", 1)[1]
                elif "</thinking>" in vl_description:
                    vl_description = vl_description.split("</thinking>", 1)[1]
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Could not connect to Qwen3-VL service: {str(exc)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Qwen-VL query failed: {str(e)}")
 
        # Step 2: Extract objects using spaCy NER & Noun phrase parsing
        objects_parsed = []
        try:
            print("Running spaCy NER and noun phrase parsing...", flush=True)
            spacy_objects = extract_objects_with_spacy(vl_description)
            for name in spacy_objects:
                objects_parsed.append({
                    "name": name,
                    "prompt": name,
                    "attributes": [],
                    "sub_objects": []
                })
        except Exception as e:
            print(f"spaCy extraction failed: {e}", flush=True)

        return {
            "success": True,
            "description": vl_description,
            "objects": objects_parsed,
            "candidates": []
        }

    else:
        # Standard description query
        active_vl_model = await get_active_model("http://qwen3-vl:8000", os.getenv("VL_MODEL", "Qwen/Qwen3-VL-8B-Thinking"))
        payload = {
            "model": active_vl_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": req.prompt},
                        {"type": media_type, **media_payload}
                    ]
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.1
        }

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post("http://qwen3-vl:8000/v1/chat/completions", json=payload, headers=headers)
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"vLLM server error: {response.text}")
                
                result = response.json()
                raw_text = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "description": raw_text,
                    "objects": []
                }
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Could not connect to Qwen3-VL service: {str(exc)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


