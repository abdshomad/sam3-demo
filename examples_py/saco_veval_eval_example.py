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
# Converted from sam3/examples/saco_veval_eval_example.ipynb
# ==========================================

import json
import os

from sam3.eval.saco_veval_eval import VEvalEvaluator

DATASETS_TO_EVAL = [
    "saco_veval_sav_test",
    "saco_veval_yt1b_test",
    "saco_veval_smartglasses_test",
]
# Update to the directory where the GT annotation and PRED files exist
GT_DIR = None # PUT YOUR ANNOTATION PATH HERE
PRED_DIR = None # PUT YOUR IMAGE PATH HERE

all_eval_res = {}
for dataset_name in DATASETS_TO_EVAL:
    gt_annot_file = os.path.join(GT_DIR, dataset_name + ".json")
    pred_file = os.path.join(PRED_DIR, dataset_name + "_preds.json")
    eval_res_file = os.path.join(PRED_DIR, dataset_name + "_eval_res.json")

    if os.path.exists(eval_res_file):
        with open(eval_res_file, "r") as f:
            eval_res = json.load(f)
    else:
        # Alternatively, we can run the evaluator offline first
        # by leveraging sam3/eval/saco_veval_eval.py
        print(f"=== Running evaluation for Pred {pred_file} vs GT {gt_annot_file} ===")
        veval_evaluator = VEvalEvaluator(
            gt_annot_file=gt_annot_file, eval_res_file=eval_res_file
        )
        eval_res = veval_evaluator.run_eval(pred_file=pred_file)
        print(f"=== Results saved to {eval_res_file} ===")

    all_eval_res[dataset_name] = eval_res

REPORT_METRICS = {
    "video_mask_demo_cgf1_micro_50_95": "cgf1",
    "video_mask_all_phrase_HOTA": "pHOTA",
}

res_to_print = []
for dataset_name in DATASETS_TO_EVAL:
    eval_res = all_eval_res[dataset_name]
    row = [dataset_name]
    for metric_k, metric_v in REPORT_METRICS.items():
        row.append(eval_res["dataset_results"][metric_k])
    res_to_print.append(row)

# Print dataset header (each dataset spans 2 metrics: 13 + 3 + 13 = 29 chars)
print("| " + " | ".join(f"{ds:^29}" for ds in DATASETS_TO_EVAL) + " |")

# Print metric header
metrics = list(REPORT_METRICS.values())
print("| " + " | ".join(f"{m:^13}" for _ in DATASETS_TO_EVAL for m in metrics) + " |")

# Print eval results
values = []
for row in res_to_print:
    values.extend([f"{v * 100:^13.1f}" for v in row[1:]])
print("| " + " | ".join(values) + " |")



