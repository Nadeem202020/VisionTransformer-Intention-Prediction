import torch
# import torch.nn as nn # Not strictly needed by this script
# import torch.nn.functional as F # Not strictly needed by this script
import numpy as np
# import pandas as pd # Not directly used unless saving results to CSV/Feather
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
# import matplotlib.pyplot as plt # Uncomment if you add plotting
from pathlib import Path
# import os # Not directly used
# import traceback # Typically for debugging, can be removed for cleaner script
import torchvision # For torchvision.ops.nms

# Project-specific imports (assuming all .py files are in the same root directory)
from constants import (GRID_HEIGHT_PX, GRID_WIDTH_PX, NUM_INTENTION_CLASSES,
                       ANCHOR_CONFIGS_PAPER, AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE,
                       INTENTIONS_MAP_REV, LIDAR_TOTAL_CHANNELS, MAP_CHANNELS) # Added missing constants
from dataset import ArgoverseIntentNetDataset, collate_fn
from model_cnn import IntentNetDetectorIntention, BasicBlock, Backbone # For CNN eval
# from model_vit import IntentNetDetectorIntentionTwoStreamViT # For ViT eval (in eval_vit.py)
from utils import generate_anchors, decode_box_predictions, apply_nms, compute_axis_aligned_iou, calculate_ap

# --- Script Configuration ---
# USER_CONFIG: Update these paths as necessary
VAL_DATA_DIR = "path/to/your/argoverse2/sensor/val" # Placeholder - User must set this
MODEL_SAVE_PATH_CNN = "./trained_models_cnn/cnn_model.pth" # Example path

# Evaluation Hyperparameters
CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.2
INFERENCE_BATCH_SIZE = 8 # Adjust based on GPU memory
NUM_WORKERS_EVAL = 0     # For DataLoader
DETECTION_IOU_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9] # For mAP calculation
IOU_THRESHOLD_FOR_INTENTION_MATCH = 0.5             # For matching detections to GT for intention eval
FEATURE_MAP_STRIDE_CNN = 8                          # Specific to CNN architecture

# Runtime Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_eval_cnn():
    """Main function to run evaluation for the CNN model."""
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Torch CUDA version: {torch.version.cuda}")
    # print(f"timm version: {timm.__version__}") # Not needed for CNN eval
    print(f"AV2 Map API Available: {AV2_MAP_AVAILABLE}")
    print(f"Shapely Available: {SHAPELY_AVAILABLE}")
    print(f"Using device for CNN evaluation: {DEVICE}")

    # --- Validate Data Path ---
    val_data_path = Path(VAL_DATA_DIR)
    if not val_data_path.is_dir():
        print(f"ERROR: Evaluation data directory not found: {VAL_DATA_DIR}")
        print("Please update the VAL_DATA_DIR variable in this script.")
        return # Changed from exit() to return for better script structure

    # --- Load Trained CNN Model ---
    print(f"\n--- Loading TRAINED CNN Model for Evaluation ---")
    print(f"Model Path: {MODEL_SAVE_PATH_CNN}")

    model_path_cnn = Path(MODEL_SAVE_PATH_CNN)
    if not model_path_cnn.is_file():
        print(f"ERROR: CNN Model checkpoint not found at {MODEL_SAVE_PATH_CNN}")
        return

    try:
        checkpoint_cnn = torch.load(MODEL_SAVE_PATH_CNN, map_location=DEVICE)
        print("CNN Checkpoint loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load CNN checkpoint from {MODEL_SAVE_PATH_CNN}: {e}")
        # traceback.print_exc() # Keep for debugging if needed
        return

    loaded_cnn_backbone_cfg = checkpoint_cnn.get('backbone_cfg')
    if not loaded_cnn_backbone_cfg:
        print("ERROR: 'backbone_cfg' not found in the CNN checkpoint. Using default (may be incorrect).")
        loaded_cnn_backbone_cfg = {
            'block': BasicBlock,
            'lidar_input_channels': LIDAR_TOTAL_CHANNELS, # From constants
            'map_input_channels': MAP_CHANNELS           # From constants
        }

    try:
        model_to_evaluate_cnn = IntentNetDetectorIntention(backbone_cfg=loaded_cnn_backbone_cfg).to(DEVICE)
        print("CNN Model structure instantiated successfully.")
    except Exception as e:
        print(f"ERROR: Failed to instantiate CNN model: {e}")
        # traceback.print_exc()
        return

    if 'model_state_dict' in checkpoint_cnn:
        try:
            model_to_evaluate_cnn.load_state_dict(checkpoint_cnn['model_state_dict'])
            print("TRAINED CNN Model state_dict loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load CNN model state_dict: {e}")
            # traceback.print_exc()
            return
    else:
        print("ERROR: 'model_state_dict' not found in the CNN checkpoint.")
        return

    model_to_evaluate_cnn.eval()
    print("TRAINED CNN Model set to evaluation mode.")
    print("--------------------")

    # --- Prepare DataLoader and Anchors for CNN Evaluation ---
    print("\n--- Preparing Evaluation DataLoader (for CNN) ---")
    try:
        eval_dataset_cnn = ArgoverseIntentNetDataset(data_dir=VAL_DATA_DIR, is_train=False)
        if not eval_dataset_cnn: # Or len(eval_dataset_cnn) == 0, though validator should raise error
            print("ERROR: Evaluation dataset is empty or failed to initialize.")
            return
        eval_loader_cnn = DataLoader(
            eval_dataset_cnn, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS_EVAL, collate_fn=collate_fn, pin_memory=(DEVICE.type == 'cuda')
        )
        print(f"Eval loader for CNN: {len(eval_loader_cnn)} batches for {len(eval_dataset_cnn)} samples.")
        if len(eval_loader_cnn) == 0 and len(eval_dataset_cnn) > 0:
            print("Warning: Eval loader for CNN is empty, but dataset is not. Check batch_size.")
    except Exception as e_data:
        print(f"Error initializing eval dataset/loader for CNN: {e_data}")
        # traceback.print_exc()
        return

    print("\n--- Generating Anchors for CNN ---")
    try:
        anchors_for_cnn_eval = generate_anchors(
            bev_height=GRID_HEIGHT_PX,
            bev_width=GRID_WIDTH_PX,
            feature_map_stride=FEATURE_MAP_STRIDE_CNN,
            anchor_configs=ANCHOR_CONFIGS_PAPER
        ).to(DEVICE)
        print(f"Anchors for CNN evaluation generated (stride {FEATURE_MAP_STRIDE_CNN}): {anchors_for_cnn_eval.shape}")
    except Exception as e_anchors:
        print(f"ERROR generating anchors for CNN: {e_anchors}")
        # traceback.print_exc()
        return

    # --- Run Inference Loop with the LOADED (TRAINED) CNN Model ---
    print("\n--- Running Inference with LOADED CNN Model ---")
    all_sample_results_cnn = []
    with torch.inference_mode(): # Changed from torch.no_grad() for more explicitness in inference
        progress_bar_cnn = tqdm(eval_loader_cnn, desc="CNN Evaluation Inference", unit="batch")
        for batch_idx, batch_data in enumerate(progress_bar_cnn):
            if batch_data is None:
                print(f"Warning: Skipping None batch {batch_idx + 1}")
                continue
            if not all(k in batch_data for k in ["lidar_bev", "map_bev", "gt_list"]):
                print(f"Warning: Skipping batch {batch_idx + 1} due to missing keys.")
                continue

            batch_size_actual = batch_data["lidar_bev"].shape[0]
            gt_list_batch_cpu = batch_data["gt_list"]

            try:
                lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
                map_bev = batch_data["map_bev"].to(DEVICE, non_blocking=True)
                det_cls_logits, det_box_preds_rel, int_logits = model_to_evaluate_cnn(lidar_bev, map_bev)

                for b_idx in range(batch_size_actual):
                    sample_preds = {
                        'pred_scores': torch.empty(0, device='cpu'),
                        'pred_boxes_xywha': torch.empty((0, 5), device='cpu'),
                        'pred_intentions': torch.empty(0, dtype=torch.long, device='cpu')
                    }
                    try:
                        cls_logits_sample = det_cls_logits[b_idx]
                        box_preds_rel_sample = det_box_preds_rel[b_idx]
                        int_logits_sample = int_logits[b_idx]
                        scores_sample = torch.sigmoid(cls_logits_sample)
                        keep_conf_indices = torch.where(scores_sample >= CONFIDENCE_THRESHOLD)[0]

                        if keep_conf_indices.numel() > 0:
                            scores_filtered = scores_sample[keep_conf_indices]
                            anchors_filtered = anchors_for_cnn_eval[keep_conf_indices]
                            box_preds_rel_filtered = box_preds_rel_sample[keep_conf_indices]
                            int_logits_filtered = int_logits_sample[keep_conf_indices]

                            boxes_decoded_abs = decode_box_predictions(box_preds_rel_filtered, anchors_filtered)
                            nms_keep_indices = apply_nms(boxes_decoded_abs, scores_filtered, NMS_IOU_THRESHOLD)

                            if nms_keep_indices.numel() > 0:
                                sample_preds['pred_scores'] = scores_filtered[nms_keep_indices].cpu()
                                sample_preds['pred_boxes_xywha'] = boxes_decoded_abs[nms_keep_indices].cpu()
                                sample_preds['pred_intentions'] = torch.argmax(int_logits_filtered[nms_keep_indices], dim=-1).cpu()
                    except Exception as e_post:
                        print(f"Error during post-processing CNN sample {b_idx} in batch {batch_idx}: {e_post}")
                        # traceback.print_exc() # Keep for debugging

                    current_gt = gt_list_batch_cpu[b_idx]
                    all_sample_results_cnn.append({
                        **sample_preds,
                        'gt_boxes_xywha': current_gt.get('boxes_xywha', torch.empty((0,5), device='cpu')),
                        'gt_intentions': current_gt.get('intentions', torch.empty(0, dtype=torch.long, device='cpu'))
                    })
            except Exception as e_batch:
                print(f"!!! ERROR processing CNN eval batch {batch_idx}: {e_batch}")
                # traceback.print_exc()

    print(f"Collected results for {len(all_sample_results_cnn)} samples using TRAINED CNN model.")

    # --- Detection Evaluation (mAP) for CNN ---
    print("\n--- Calculating Detection mAP for TRAINED CNN Model ---")
    average_precisions_per_iou_cnn = {iou_thresh: [] for iou_thresh in DETECTION_IOU_THRESHOLDS}
    if not all_sample_results_cnn:
        print("WARNING: 'all_sample_results_cnn' is empty. Cannot calculate mAP.")
    else:
        for sample_result in tqdm(all_sample_results_cnn, desc="Calculating AP per sample (CNN)", unit="sample"):
            pred_scores = sample_result['pred_scores']
            pred_boxes = sample_result['pred_boxes_xywha']
            gt_boxes = sample_result['gt_boxes_xywha']
            num_gt = gt_boxes.shape[0]
            num_pred = pred_boxes.shape[0]

            for iou_thresh_eval in DETECTION_IOU_THRESHOLDS:
                if num_pred == 0:
                    average_precisions_per_iou_cnn[iou_thresh_eval].append(1.0 if num_gt == 0 else 0.0)
                    continue
                if num_gt == 0: # Has predictions but no GT
                    average_precisions_per_iou_cnn[iou_thresh_eval].append(0.0)
                    continue

                # Sort predictions by score
                sort_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes_sorted = pred_boxes[sort_idx]
                # pred_scores_sorted = pred_scores[sort_idx] # Not used directly in AP calc below but good for debug

                iou_matrix = compute_axis_aligned_iou(pred_boxes_sorted[:, :4].float(), gt_boxes[:, :4].float())
                gt_matched_flags = torch.zeros(num_gt, dtype=torch.bool, device=DEVICE)
                tp_flags = torch.zeros(num_pred, dtype=torch.bool, device=DEVICE) 

                for pred_idx in range(num_pred):
                    pred_ious_with_gts = iou_matrix[pred_idx, :]
                    if pred_ious_with_gts.numel() == 0: continue # Should not happen if num_gt > 0

                    best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                    if best_iou_for_pred >= iou_thresh_eval:
                        if not gt_matched_flags[best_gt_idx_for_pred]:
                            tp_flags[pred_idx] = True
                            gt_matched_flags[best_gt_idx_for_pred] = True
                        # else: # Already matched to this GT by a higher-scoring prediction
                            # fp_flags[pred_idx] = True # This pred is a FP for this GT
                    # else: # Does not meet IoU threshold
                        # fp_flags[pred_idx] = True # This pred is a FP

                # Calculate AP using 11-point interpolation or other standard method
                # The following is a common way to calculate AP:
                tp_cumsum = torch.cumsum(tp_flags.float(), dim=0)
                # fp_cumsum = torch.cumsum(fp_flags.float(), dim=0) # Not needed if using (pred_idx + 1) for total preds
                recall_steps = tp_cumsum / (num_gt + 1e-9) # Add epsilon to avoid division by zero
                precision_steps = tp_cumsum / (torch.arange(1, num_pred + 1, device=tp_flags.device).float() + 1e-9) # tp / (tp+fp)

                ap = calculate_ap(recall_steps.cpu().numpy(), precision_steps.cpu().numpy())
                average_precisions_per_iou_cnn[iou_thresh_eval].append(ap)

    print("\n--- TRAINED CNN Detection Results (mAP) ---")
    for iou_thresh_eval, ap_list in average_precisions_per_iou_cnn.items():
        mean_ap_cnn = np.mean(ap_list) if ap_list else 0.0
        print(f"TRAINED CNN mAP @ IoU={iou_thresh_eval:.1f}: {mean_ap_cnn:.4f}")

    # --- Intention Prediction Evaluation for CNN ---
    print("\n--- Calculating Intention Prediction Metrics for TRAINED CNN Model ---")
    matched_pred_intentions_cnn = []
    matched_gt_intentions_cnn = []
    if not all_sample_results_cnn:
        print("WARNING: 'all_sample_results_cnn' is empty. Cannot calculate intention metrics.")
    else:
        for sample_result in tqdm(all_sample_results_cnn, desc="Matching for Intention Eval (CNN)", unit="sample"):
            pred_scores = sample_result['pred_scores']
            pred_boxes = sample_result['pred_boxes_xywha']
            pred_intentions = sample_result['pred_intentions']
            gt_boxes = sample_result['gt_boxes_xywha']
            gt_intentions = sample_result['gt_intentions']

            num_gt = gt_boxes.shape[0]
            num_pred_after_nms = pred_boxes.shape[0]

            if num_gt == 0 or num_pred_after_nms == 0:
                continue

            iou_matrix_intent = compute_axis_aligned_iou(pred_boxes[:, :4].float(), gt_boxes[:, :4].float())
            gt_matched_for_intent = torch.zeros(num_gt, dtype=torch.bool)

            # Sort predictions by score for consistent matching (highest score gets first dibs)
            sort_idx_intent = torch.argsort(pred_scores, descending=True)

            for i in range(num_pred_after_nms):
                pred_idx = sort_idx_intent[i] # Process in order of score
                pred_ious_with_gts = iou_matrix_intent[pred_idx, :]
                if pred_ious_with_gts.numel() == 0: continue

                best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                if best_iou_for_pred >= IOU_THRESHOLD_FOR_INTENTION_MATCH:
                    if not gt_matched_for_intent[best_gt_idx_for_pred]:
                        gt_matched_for_intent[best_gt_idx_for_pred] = True
                        matched_pred_intentions_cnn.append(pred_intentions[pred_idx].item())
                        matched_gt_intentions_cnn.append(gt_intentions[best_gt_idx_for_pred].item())
                        # Once a GT is matched, it cannot be matched again by a lower-scoring prediction

    if matched_pred_intentions_cnn:
        print(f"\n--- TRAINED CNN Intention Prediction Results (on TP detections @ IoU>={IOU_THRESHOLD_FOR_INTENTION_MATCH}) ---")
        labels_report = list(range(NUM_INTENTION_CLASSES))
        # Ensure INTENTIONS_MAP_REV is available (from constants.py)
        target_names_report = [INTENTIONS_MAP_REV.get(i, f"Class_{i}") for i in labels_report]

        overall_acc_cnn = accuracy_score(matched_gt_intentions_cnn, matched_pred_intentions_cnn)
        print(f"TRAINED CNN Overall Accuracy: {overall_acc_cnn:.4f}")

        f1_macro_cnn = f1_score(matched_gt_intentions_cnn, matched_pred_intentions_cnn, labels=labels_report, average='macro', zero_division=0)
        f1_weighted_cnn = f1_score(matched_gt_intentions_cnn, matched_pred_intentions_cnn, labels=labels_report, average='weighted', zero_division=0)
        print(f"TRAINED CNN F1 (Macro):   {f1_macro_cnn:.4f}")
        print(f"TRAINED CNN F1 (Weighted): {f1_weighted_cnn:.4f}")

        print("TRAINED CNN F1 (Per Class):")
        f1_per_class_cnn = f1_score(matched_gt_intentions_cnn, matched_pred_intentions_cnn, labels=labels_report, average=None, zero_division=0)
        for i, name in enumerate(target_names_report):
            print(f"  {name:<20}: {f1_per_class_cnn[i]:.4f}")
    else:
        print(f"\nNo True Positive detections found for TRAINED CNN model at IoU >= {IOU_THRESHOLD_FOR_INTENTION_MATCH} to evaluate intention prediction.")

    print("\n--- Evaluation Script for TRAINED CNN Finished ---")

if __name__ == '__main__':
    main_eval_cnn()