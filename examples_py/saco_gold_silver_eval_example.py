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
# Converted from sam3/examples/saco_gold_silver_eval_example.ipynb
# ==========================================

# Copyright (c) Meta Platforms, Inc. and affiliates.

import copy
import json
import os

import numpy as np

from pycocotools.coco import COCO
from sam3.eval.cgf1_eval import CGF1Evaluator

# Update to the directory where the GT annotation and PRED files exist
GT_DIR = # PUT YOUR PATH HERE
PRED_DIR = # PUT YOUR PATH HERE

# Relative file names for GT files for 7 SA-Co/Gold subsets
saco_gold_gts = {
    # MetaCLIP Captioner
    "metaclip_nps": [
            "gold_metaclip_merged_a_release_test.json",
            "gold_metaclip_merged_b_release_test.json",
            "gold_metaclip_merged_c_release_test.json",
    ],
    # SA-1B captioner
    "sa1b_nps": [
            "gold_sa1b_merged_a_release_test.json",
            "gold_sa1b_merged_b_release_test.json",
            "gold_sa1b_merged_c_release_test.json",
    ],
    # Crowded
    "crowded": [
            "gold_crowded_merged_a_release_test.json",
            "gold_crowded_merged_b_release_test.json",
            "gold_crowded_merged_c_release_test.json",
    ],
    # FG Food
    "fg_food": [
            "gold_fg_food_merged_a_release_test.json",
            "gold_fg_food_merged_b_release_test.json",
            "gold_fg_food_merged_c_release_test.json",
    ],
    # FG Sports
    "fg_sports_equipment": [
            "gold_fg_sports_equipment_merged_a_release_test.json",
            "gold_fg_sports_equipment_merged_b_release_test.json",
            "gold_fg_sports_equipment_merged_c_release_test.json",
    ],
    # Attributes
    "attributes": [
            "gold_attributes_merged_a_release_test.json",
            "gold_attributes_merged_b_release_test.json",
            "gold_attributes_merged_c_release_test.json",
    ],
    # Wiki common
    "wiki_common": [
            "gold_wiki_common_merged_a_release_test.json",
            "gold_wiki_common_merged_b_release_test.json",
            "gold_wiki_common_merged_c_release_test.json",
    ],
}

results_gold = {}
results_gold_bbox = {}

for subset_name, gts in saco_gold_gts.items():
    print("Processing subset: ", subset_name)
    gt_paths = [os.path.join(GT_DIR, gt) for gt in gts]
    pred_path = os.path.join(PRED_DIR, f"gold_{subset_name}/dumps/gold_{subset_name}/coco_predictions_segm.json")
    
    evaluator = CGF1Evaluator(gt_path=gt_paths, verbose=True, iou_type="segm") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_segm_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_segm_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_segm_positive_micro_F1"] * 100
    results_gold[subset_name] = cur_results

    # Also eval bbox    
    evaluator = CGF1Evaluator(gt_path=gt_paths, verbose=True, iou_type="bbox") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_bbox_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_bbox_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_bbox_positive_micro_F1"] * 100
    results_gold_bbox[subset_name] = cur_results

# Compute averages
METRICS = ["cgf1", "il_mcc", "pmf1"]
avg_stats, avg_stats_bbox = {}, {}
for key in METRICS:
    avg_stats[key] = sum(res[key] for res in results_gold.values()) / len(results_gold)
    avg_stats_bbox[key] = sum(res[key] for res in results_gold_bbox.values()) / len(results_gold_bbox)
results_gold["Average"] = avg_stats
results_gold_bbox["Average"] = avg_stats_bbox

# Pretty print segmentation results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_gold:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_gold[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))

# Pretty print bbox detection results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_gold:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_gold_bbox[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))

# Update to the directory where the GT annotation and PRED files exist
GT_DIR =  # PUT YOUR PATH HERE
PRED_DIR =  # PUT YOUR PATH HERE

saco_silver_gts = {
    "bdd100k": "silver_bdd100k_merged_test.json",
    "droid": "silver_droid_merged_test.json",
    "ego4d": "silver_ego4d_merged_test.json",
    "food_rec": "silver_food_rec_merged_test.json",
    "geode": "silver_geode_merged_test.json",
    "inaturalist": "silver_inaturalist_merged_test.json",
    "nga_art": "silver_nga_art_merged_test.json",
    "sav": "silver_sav_merged_test.json",
    "yt1b": "silver_yt1b_merged_test.json",
    "fathomnet": "silver_fathomnet_test.json",
}

results_silver = {}
results_silver_bbox = {}

for subset_name, gt in saco_silver_gts.items():
    print("Processing subset: ", subset_name)
    gt_path = os.path.join(GT_DIR, gt)
    pred_path = os.path.join(PRED_DIR, f"silver_{subset_name}/dumps/silver_{subset_name}/coco_predictions_segm.json")
    
    evaluator = CGF1Evaluator(gt_path=gt_path, verbose=True, iou_type="segm") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_segm_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_segm_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_segm_positive_micro_F1"] * 100
    results_silver[subset_name] = cur_results

    # Also eval bbox    
    evaluator = CGF1Evaluator(gt_path=gt_path, verbose=True, iou_type="bbox") 
    summary = evaluator.evaluate(pred_path)
    print(summary)

    cur_results = {}
    cur_results["cgf1"] = summary["cgF1_eval_bbox_cgF1"] * 100
    cur_results["il_mcc"] = summary["cgF1_eval_bbox_IL_MCC"]
    cur_results["pmf1"] = summary["cgF1_eval_bbox_positive_micro_F1"] * 100
    results_silver_bbox[subset_name] = cur_results

# Compute averages
METRICS = ["cgf1", "il_mcc", "pmf1"]
avg_stats, avg_stats_bbox = {}, {}
for key in METRICS:
    avg_stats[key] = sum(res[key] for res in results_silver.values()) / len(results_silver)
    avg_stats_bbox[key] = sum(res[key] for res in results_silver_bbox.values()) / len(results_silver_bbox)
results_silver["Average"] = avg_stats
results_silver_bbox["Average"] = avg_stats_bbox

# Pretty print segmentation results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_silver:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_silver[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))

# Pretty print bbox detection results
from IPython.display import HTML, display

row1, row2, row3 = "", "", ""
for subset in results_silver_bbox:
    row1 += f'<th colspan="3" style="text-align:center;border-left-style:solid;border-left-width:1px">{subset}</th>'
    row2 += "<th style='border-left-style:solid;border-left-width:1px'>" + "</th><th>".join(METRICS) + "</th>"
    row3 += "<td style='border-left-style:solid;border-left-width:1px'>" + "</td><td>".join([str(round(results_silver_bbox[subset][k], 2)) for k in METRICS])  + "</td>"

display(HTML(
   f"<table><thead><tr>{row1}</tr><tr>{row2}</tr></thead><tbody><tr>{row3}</tr></tbody></table>"
))



