import os
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


# ==========================================
# Converted from sam3/examples/saco_veval_vis_example.ipynb
# ==========================================

# Copyright (c) Meta Platforms, Inc. and affiliates.

using_colab = False

if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
#     !{sys.executable} -m pip install opencv-python matplotlib scikit-learn
#     !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/sam3.git'

import os
from glob import glob

import numpy as np
import utils

from matplotlib import pyplot as plt

COLORS = utils.pascal_color_map()[1:]

# Preapre the data path
DATA_DIR = "./sam3_saco_veval_data" # PUT YOUR DATA PATH HERE
ANNOT_DIR = os.path.join(DATA_DIR, "annotation")

# Load the SACO/Veval annotation files
annot_file_list = glob(os.path.join(ANNOT_DIR, "*veval*.json"))
annot_dfs = utils.get_annot_dfs(file_list=annot_file_list)

annot_dfs.keys()

annot_dfs["saco_veval_yt1b_val"].keys()

annot_dfs["saco_veval_yt1b_val"]["info"]

annot_dfs["saco_veval_yt1b_val"]["videos"].head(3)

annot_dfs["saco_veval_yt1b_val"]["annotations"].head(3)

annot_dfs["saco_veval_yt1b_val"]["categories"].head(3)

annot_dfs["saco_veval_yt1b_val"]["video_np_pairs"].head(3)

# Select a target dataset
target_dataset_name = "saco_veval_yt1b_val"

# visualize a random positive video-np pair
df_pairs = annot_dfs[target_dataset_name]["video_np_pairs"]
df_positive_pairs = df_pairs[df_pairs.num_masklets > 0]
rand_idx = np.random.randint(len(df_positive_pairs))
pair_row = df_positive_pairs.iloc[rand_idx]
video_id = pair_row.video_id
noun_phrase = pair_row.noun_phrase
print(f"Randomly selected video-np pair: video_id={video_id}, noun_phrase={noun_phrase}")

def display_image_in_subplot(img, axes, row, col, title=""):
    axes[row, col].imshow(img)
    axes[row, col].set_title(title)
    axes[row, col].axis('off')

num_frames_to_show = 5  # Number of frames to show per dataset
every_n_frames = 4  # Interval between frames to show

fig, axes = plt.subplots(num_frames_to_show, 3, figsize=(15, 5 * num_frames_to_show))

for idx in range(0, num_frames_to_show):
    sampled_frame_idx = idx * every_n_frames
    print(f"Reading annotations for frame {sampled_frame_idx}")
    # Get the frame and the corresponding masks and noun phrases
    frame, annot_masks, annot_noun_phrases = utils.get_all_annotations_for_frame(
        annot_dfs[target_dataset_name], video_id=video_id, frame_idx=sampled_frame_idx, data_dir=DATA_DIR, dataset=target_dataset_name
    )
    # Filter masks and noun phrases by the selected noun phrase
    annot_masks = [m for m, np in zip(annot_masks, annot_noun_phrases) if np == noun_phrase]

    # Show the frame
    display_image_in_subplot(frame, axes, idx, 0, f"{target_dataset_name} - {noun_phrase} - Frame {sampled_frame_idx}")

    # Show the annotated masks
    if annot_masks is None:
        print(f"No masks found for video_id {video_id} at frame {sampled_frame_idx}")
    else:
        # Show all masks over a white background
        all_masks = utils.draw_masks_to_frame(
            frame=np.ones_like(frame)*255, masks=annot_masks, colors=COLORS[: len(annot_masks)]
        )
        display_image_in_subplot(all_masks, axes, idx, 1, f"{target_dataset_name} - {noun_phrase} - Frame {sampled_frame_idx} - Masks")
        
        # Show masks overlaid on the frame
        masked_frame = utils.draw_masks_to_frame(
            frame=frame, masks=annot_masks, colors=COLORS[: len(annot_masks)]
        )
        display_image_in_subplot(masked_frame, axes, idx, 2, f"Dataset: {target_dataset_name} - {noun_phrase} - Frame {sampled_frame_idx} - Masks overlaid")

plt.tight_layout()
plt.show()



