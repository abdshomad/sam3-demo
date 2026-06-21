import json
import glob
import os

# Create target directory for python scripts
os.makedirs("examples_py", exist_ok=True)

# Boilerplate code to prepend to all converted scripts
boilerplate = """import os
import sys

# Prepend the workspace and sam3 directories to sys.path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(current_dir)
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)
sam3_root = os.path.join(workspace_dir, "sam3")
if sam3_root not in sys.path:
    sys.path.insert(0, sam3_root)

# Set Hugging Face home and token
os.environ["HF_HOME"] = os.path.join(workspace_dir, ".hf_cache")
env_path = os.path.join(workspace_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
                break

# Configure matplotlib for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Monkey patch plt.show to save figures to the results folder
_fig_count = 0
_orig_show = plt.show

def _patched_show(*args, **kwargs):
    global _fig_count
    script_name = os.path.basename(sys.argv[0]).replace(".py", "")
    out_dir = os.path.join(workspace_dir, "results/examples_outputs")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{script_name}_fig_{_fig_count}.png")
    try:
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved example figure to {fig_path}")
    except Exception as e:
        print(f"Could not save example figure: {e}")
    _fig_count += 1
    return _orig_show(*args, **kwargs)

plt.show = _patched_show

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

# Configure PyTorch memory settings to be robust on shared GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""

notebooks = glob.glob("sam3/examples/*.ipynb")
print(f"Found {len(notebooks)} notebooks to convert.")

for notebook_path in notebooks:
    filename = os.path.basename(notebook_path)
    py_filename = filename.replace(".ipynb", ".py")
    py_path = os.path.join("examples_py", py_filename)
    
    with open(notebook_path, "r", encoding="utf-8") as f:
        try:
            notebook = json.load(f)
        except Exception as e:
            print(f"Error parsing {notebook_path}: {e}")
            continue
            
    py_lines = [boilerplate, "\n\n# ==========================================\n"]
    py_lines.append(f"# Converted from {notebook_path}\n# ==========================================\n\n")
    
    for cell in notebook.get("cells", []):
        cell_type = cell.get("cell_type")
        if cell_type == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                code = "".join(source)
            else:
                code = source
                
            clean_code_lines = []
            for line in code.splitlines():
                stripped = line.strip()
                # Comment out Jupyter magic commands
                if stripped.startswith("%") or stripped.startswith("!"):
                    clean_code_lines.append(f"# {line}")
                else:
                    clean_code_lines.append(line)
            py_lines.append("\n".join(clean_code_lines))
            py_lines.append("\n\n")
            
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("".join(py_lines))
    print(f"Successfully converted {notebook_path} -> {py_path}")
