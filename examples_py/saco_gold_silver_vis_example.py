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
# Converted from sam3/examples/saco_gold_silver_vis_example.ipynb
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
import sam3.visualization_utils as utils

from matplotlib import pyplot as plt

COLORS = utils.pascal_color_map()[1:]

# Preapre the data path
ANNOT_DIR = None # PUT YOUR ANNOTATION PATH HERE
IMG_DIR = None # PUT YOUR IMAGE PATH HERE

# Load the SA-CO/Gold annotation files
annot_file_list = glob(os.path.join(ANNOT_DIR, "*gold*.json"))
annot_dfs = utils.get_annot_dfs(file_list=annot_file_list)

annot_dfs.keys()

annot_dfs["gold_fg_sports_equipment_merged_a_release_test"].keys()

annot_dfs["gold_fg_sports_equipment_merged_a_release_test"]["info"]

annot_dfs["gold_fg_sports_equipment_merged_a_release_test"]["images"].head(3)

annot_dfs["gold_fg_sports_equipment_merged_a_release_test"]["annotations"].head(3)

# Select a target dataset
target_dataset_name = "gold_fg_food_merged_a_release_test"

import cv2
from pycocotools import mask as mask_util
from collections import defaultdict

# Group GT annotations by image_id
gt_image_np_pairs = annot_dfs[target_dataset_name]["images"]
gt_annotations = annot_dfs[target_dataset_name]["annotations"]

gt_image_np_map = {img["id"]: img for _, img in gt_image_np_pairs.iterrows()}
gt_image_np_ann_map = defaultdict(list)
for _, ann in gt_annotations.iterrows():
    image_id = ann["image_id"]
    if image_id not in gt_image_np_ann_map:
        gt_image_np_ann_map[image_id] = []
    gt_image_np_ann_map[image_id].append(ann)

positiveNPs = common_image_ids = [img_id for img_id in gt_image_np_map.keys() if img_id in gt_image_np_ann_map and gt_image_np_ann_map[img_id]]
negativeNPs = [img_id for img_id in gt_image_np_map.keys() if img_id not in gt_image_np_ann_map or not gt_image_np_ann_map[img_id]]

num_image_nps_to_show = 10
fig, axes = plt.subplots(num_image_nps_to_show, 3, figsize=(15, 5 * num_image_nps_to_show))
for idx in range(num_image_nps_to_show):
    rand_idx = np.random.randint(len(positiveNPs))
    image_id = positiveNPs[rand_idx]
    noun_phrase = gt_image_np_map[image_id]["text_input"]
    img_rel_path = gt_image_np_map[image_id]["file_name"]
    full_path = os.path.join(IMG_DIR, f"{img_rel_path}")
    img = cv2.imread(full_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt_annotation = gt_image_np_ann_map[image_id]

    def display_image_in_subplot(img, axes, row, col, title=""):
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')


    noun_phrases = [noun_phrase]
    annot_masks = [mask_util.decode(ann["segmentation"]) for ann in gt_annotation]

    # Show the image
    display_image_in_subplot(img, axes, idx, 0, f"{noun_phrase}")

    # Show all masks over a white background
    all_masks = utils.draw_masks_to_frame(
        frame=np.ones_like(img)*255, masks=annot_masks, colors=COLORS[: len(annot_masks)]
    )
    display_image_in_subplot(all_masks, axes, idx, 1, f"{noun_phrase} - Masks only")

    # Show masks overlaid on the image
    masked_frame = utils.draw_masks_to_frame(
        frame=img, masks=annot_masks, colors=COLORS[: len(annot_masks)]
    )
    display_image_in_subplot(masked_frame, axes, idx, 2, f"{noun_phrase} - Masks overlaid")



