import torch
# import torch.nn as nn # Optional, not directly used
# import torch.nn.functional as F # Optional
import numpy as np
# import pandas as pd # Optional, if not saving results to DataFrame
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
# import matplotlib.pyplot as plt # Optional, if adding plotting
from pathlib import Path
import timm # For timm version print, and potentially if backbone_cfg needs it
import torchvision # For torchvision.ops.nms

# Project-specific imports (assuming all .py files are in the same root directory)
from constants import (GRID_HEIGHT_PX, GRID_WIDTH_PX, NUM_INTENTION_CLASSES, ANCHOR_CONFIGS_PAPER,
                       AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE, INTENTIONS_MAP_REV,
                       DETECTION_IOU_THRESHOLDS, IOU_THRESHOLD_FOR_INTENTION_MATCH,
                       LIDAR_TOTAL_CHANNELS, MAP_CHANNELS) # Added constants needed for default config
from dataset import ArgoverseIntentNetDataset, collate_fn
from model_vit import IntentNetViT # Correct ViT model import
from utils import generate_anchors, decode_box_predictions, apply_nms, compute_axis_aligned_iou, calculate_ap

# --- Script Configuration ---
# USER_CONFIG: Update this path to your Argoverse 2 validation data
VAL_DATA_DIR = "./data/argoverse2/sensor/val"  # Example placeholder
# USER_CONFIG: Path to your TRAINED ViT model checkpoint
MODEL_SAVE_PATH_VIT = "./trained_models_vit/vit_model.pth" # Example (ensure consistent with train_vit.py)

# Evaluation Hyperparameters
CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.2
INFERENCE_BATCH_SIZE = 8
NUM_WORKERS_EVAL = 0
# FEATURE_MAP_STRIDE_VIT will be determined from the loaded_backbone_cfg

# Runtime Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_eval_vit():
    """Main function to run evaluation for the Vision Transformer model."""
    print(f"--- ViT Model Evaluation ---")
    print(f"Torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Torch CUDA version: {torch.version.cuda}")
    if hasattr(timm, '__version__'): # Check if timm has __version__
        print(f"timm version: {timm.__version__}")
    print(f"Using device: {DEVICE}")
    print(f"AV2 Map API Available: {AV2_MAP_AVAILABLE}")
    print(f"Shapely Available: {SHAPELY_AVAILABLE}")

    val_data_path = Path(VAL_DATA_DIR)
    if not val_data_path.is_dir():
        print(f"ERROR: Evaluation data directory not found: {VAL_DATA_DIR}")
        print(f"Please update the VAL_DATA_DIR variable in {__file__}.")
        return

    model_path_vit = Path(MODEL_SAVE_PATH_VIT)
    if not model_path_vit.is_file():
        print(f"ERROR: ViT Model checkpoint not found at {MODEL_SAVE_PATH_VIT}")
        print(f"Please update the MODEL_SAVE_PATH_VIT variable in {__file__} or train the model first.")
        return

    # --- Load Trained ViT Model ---
    print(f"\nLoading TRAINED ViT Model from: {MODEL_SAVE_PATH_VIT}")
    try:
        checkpoint_vit = torch.load(MODEL_SAVE_PATH_VIT, map_location=DEVICE)
        print("ViT Checkpoint loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load ViT checkpoint from {MODEL_SAVE_PATH_VIT}: {e}")
        return

    loaded_vit_backbone_cfg = checkpoint_vit.get('backbone_cfg')
    if not loaded_vit_backbone_cfg:
        print("ERROR: 'backbone_cfg' not found in ViT checkpoint. Cannot reliably instantiate model or anchors.")
        return

    try:
        # Ensure 'img_size' is in the loaded config for TwoStreamViTBackbone
        loaded_vit_backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
        # Set other defaults if they might be missing and are critical for TwoStreamViTBackbone
        loaded_vit_backbone_cfg.setdefault('lidar_input_channels', LIDAR_TOTAL_CHANNELS)
        loaded_vit_backbone_cfg.setdefault('map_input_channels', MAP_CHANNELS)
        loaded_vit_backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch16_224') # Match training
        loaded_vit_backbone_cfg.setdefault('vit_model_name_map', 'vit_tiny_patch16_224')   # Match training
        loaded_vit_backbone_cfg.setdefault('pretrained_lidar', False) # Match training
        loaded_vit_backbone_cfg.setdefault('pretrained_map', False)   # Match training
        loaded_vit_backbone_cfg.setdefault('drop_path_rate_lidar', 0.1)
        loaded_vit_backbone_cfg.setdefault('drop_path_rate_map', 0.1)
        loaded_vit_backbone_cfg.setdefault('lidar_adapter_out_channels', 192) # Match training
        loaded_vit_backbone_cfg.setdefault('map_adapter_out_channels', 128)   # Match training


        model_to_evaluate_vit = IntentNetViT(backbone_cfg=loaded_vit_backbone_cfg).to(DEVICE)
        print("ViT Model structure instantiated successfully using loaded configuration.")
        model_to_evaluate_vit.load_state_dict(checkpoint_vit['model_state_dict'])
        model_to_evaluate_vit.eval()
        print("TRAINED ViT Model state_dict loaded and model set to evaluation mode.")
    except Exception as e:
        print(f"ERROR: Failed to instantiate or load state_dict for ViT model: {e}")
        return
    print("--------------------")

    # --- Prepare DataLoader and Anchors ---
    print("\nPreparing Evaluation DataLoader...")
    try:
        eval_dataset_vit = ArgoverseIntentNetDataset(data_dir=VAL_DATA_DIR, is_train=False)
        if len(eval_dataset_vit) == 0:
            print("ERROR: Evaluation dataset is empty. Check data path and scenario validation.")
            return
        eval_loader_vit = DataLoader(
            eval_dataset_vit, batch_size=INFERENCE_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS_EVAL, collate_fn=collate_fn, pin_memory=(DEVICE.type == 'cuda')
        )
        print(f"Eval DataLoader: {len(eval_loader_vit)} batches for {len(eval_dataset_vit)} samples.")
    except Exception as e_data:
        print(f"Error initializing evaluation dataset/loader: {e_data}")
        return

    print("\nGenerating Anchors for ViT...")
    try:
        vit_model_name_for_stride = loaded_vit_backbone_cfg.get('vit_model_name_lidar', 'vit_small_patch16_224')
        eval_vit_fm_stride = int(vit_model_name_for_stride.split('_patch')[-1].split('_')[0])
        print(f"Using feature_map_stride = {eval_vit_fm_stride} for ViT anchor generation (from loaded config).")

        anchors_for_vit_eval = generate_anchors(
            bev_height=GRID_HEIGHT_PX,
            bev_width=GRID_WIDTH_PX,
            feature_map_stride=eval_vit_fm_stride,
            anchor_configs=ANCHOR_CONFIGS_PAPER
        ).to(DEVICE)
        print(f"Anchors for ViT evaluation generated (stride {eval_vit_fm_stride}), shape: {anchors_for_vit_eval.shape}")
    except Exception as e_anchors:
        print(f"ERROR generating anchors for ViT: {e_anchors}")
        return
    print("--------------------")

    # --- Run Inference ---
    print("\nRunning Inference with ViT Model...")
    all_sample_results_vit = []
    with torch.inference_mode():
        progress_bar_vit = tqdm(eval_loader_vit, desc="ViT Evaluation Inference", unit="batch")
        for batch_data in progress_bar_vit:
            if batch_data is None: continue
            batch_size_actual = batch_data["lidar_bev"].shape[0]
            gt_list_batch_cpu = batch_data["gt_list"]

            try:
                lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
                map_bev = batch_data["map_bev"].to(DEVICE, non_blocking=True)
                det_cls_logits, det_box_preds_rel, intent_logits = model_to_evaluate_vit(lidar_bev, map_bev)

                for b_idx in range(batch_size_actual):
                    sample_preds = {
                        'pred_scores': torch.empty(0, device='cpu'),
                        'pred_boxes_xywha': torch.empty((0, 5), device='cpu'),
                        'pred_intentions': torch.empty(0, dtype=torch.long, device='cpu')
                    }
                    try:
                        cls_logits_sample = det_cls_logits[b_idx]
                        box_preds_rel_sample = det_box_preds_rel[b_idx]
                        int_logits_sample = intent_logits[b_idx]
                        scores_sample = torch.sigmoid(cls_logits_sample)

                        keep_conf_indices = torch.where(scores_sample >= CONFIDENCE_THRESHOLD)[0]
                        if keep_conf_indices.numel() > 0:
                            scores_filtered = scores_sample[keep_conf_indices]
                            anchors_filtered = anchors_for_vit_eval[keep_conf_indices]
                            box_preds_rel_filtered = box_preds_rel_sample[keep_conf_indices]
                            int_logits_filtered = int_logits_sample[keep_conf_indices]

                            boxes_decoded_abs = decode_box_predictions(box_preds_rel_filtered, anchors_filtered)
                            nms_keep_indices = apply_nms(boxes_decoded_abs, scores_filtered, NMS_IOU_THRESHOLD)

                            if nms_keep_indices.numel() > 0:
                                sample_preds['pred_scores'] = scores_filtered[nms_keep_indices].cpu()
                                sample_preds['pred_boxes_xywha'] = boxes_decoded_abs[nms_keep_indices].cpu()
                                sample_preds['pred_intentions'] = torch.argmax(int_logits_filtered[nms_keep_indices], dim=-1).cpu()
                    except Exception as e_post:
                        print(f"Error during post-processing ViT sample in batch: {e_post}")

                    current_gt = gt_list_batch_cpu[b_idx]
                    all_sample_results_vit.append({
                        **sample_preds,
                        'gt_boxes_xywha': current_gt.get('boxes_xywha', torch.empty((0,5), device='cpu')),
                        'gt_intentions': current_gt.get('intentions', torch.empty(0, dtype=torch.long, device='cpu'))
                    })
            except Exception as e_batch:
                print(f"ERROR processing ViT evaluation batch: {e_batch}")
    print(f"Collected results for {len(all_sample_results_vit)} samples.")

    # --- Detection Evaluation (mAP) ---
    print("\n--- Calculating Detection mAP (ViT Model) ---")
    average_precisions_per_iou_vit = {iou_thresh: [] for iou_thresh in DETECTION_IOU_THRESHOLDS}
    if not all_sample_results_vit:
        print("WARNING: No sample results to calculate mAP for ViT.")
    else:
        for sample_result in tqdm(all_sample_results_vit, desc="Calculating AP per sample (ViT)", unit="sample"):
            pred_scores = sample_result['pred_scores']
            pred_boxes = sample_result['pred_boxes_xywha']
            gt_boxes = sample_result['gt_boxes_xywha']
            num_gt = gt_boxes.shape[0]
            num_pred = pred_boxes.shape[0]

            for iou_thresh_eval in DETECTION_IOU_THRESHOLDS:
                if num_pred == 0:
                    average_precisions_per_iou_vit[iou_thresh_eval].append(1.0 if num_gt == 0 else 0.0)
                    continue
                if num_gt == 0:
                    average_precisions_per_iou_vit[iou_thresh_eval].append(0.0)
                    continue

                sort_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes_sorted = pred_boxes[sort_idx]

                iou_matrix = compute_axis_aligned_iou(pred_boxes_sorted[:, :4].float(), gt_boxes[:, :4].float())
                gt_matched_flags = torch.zeros(num_gt, dtype=torch.bool, device=DEVICE)
                tp_flags = torch.zeros(num_pred, dtype=torch.bool, device=DEVICE) 

                for pred_idx_sorted in range(num_pred): # Iterate over sorted predictions
                    pred_ious_with_gts = iou_matrix[pred_idx_sorted, :]
                    if pred_ious_with_gts.numel() == 0: continue
                    best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                    if best_iou_for_pred >= iou_thresh_eval:
                        if not gt_matched_flags[best_gt_idx_for_pred]:
                            tp_flags[pred_idx_sorted] = True # Mark this sorted prediction as TP
                            gt_matched_flags[best_gt_idx_for_pred] = True
                
                tp_cumsum = torch.cumsum(tp_flags.float(), dim=0)
                recall_steps = tp_cumsum / (num_gt + 1e-9)
                precision_steps = tp_cumsum / (torch.arange(1, num_pred + 1, device=tp_flags.device).float() + 1e-9) # Denominator is num_pred_so_far

                ap = calculate_ap(recall_steps.cpu().numpy(), precision_steps.cpu().numpy())
                average_precisions_per_iou_vit[iou_thresh_eval].append(ap)

    print("\n--- ViT Detection Results (mAP) ---")
    for iou_thresh_eval, ap_list in average_precisions_per_iou_vit.items():
        mean_ap_vit = np.mean(ap_list) if ap_list else 0.0
        print(f"ViT mAP @ IoU={iou_thresh_eval:.1f}: {mean_ap_vit:.4f}")

    # --- Intention Prediction Evaluation ---
    print("\n--- Calculating Intention Prediction Metrics (ViT Model) ---")
    matched_pred_intentions_vit = []
    matched_gt_intentions_vit = []
    if not all_sample_results_vit:
        print("WARNING: No sample results to calculate intention metrics for ViT.")
    else:
        for sample_result in tqdm(all_sample_results_vit, desc="Matching for Intention Eval (ViT)", unit="sample"):
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
            
            sort_idx_intent = torch.argsort(pred_scores, descending=True)

            for i in range(num_pred_after_nms):
                pred_original_idx = sort_idx_intent[i] # Original index of this prediction in the NMS'd list
                
                pred_ious_with_gts = iou_matrix_intent[pred_original_idx, :]
                if pred_ious_with_gts.numel() == 0: continue
                best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                if best_iou_for_pred >= IOU_THRESHOLD_FOR_INTENTION_MATCH:
                    if not gt_matched_for_intent[best_gt_idx_for_pred]:
                        gt_matched_for_intent[best_gt_idx_for_pred] = True
                        matched_pred_intentions_vit.append(pred_intentions[pred_original_idx].item())
                        matched_gt_intentions_vit.append(gt_intentions[best_gt_idx_for_pred].item())

    if matched_pred_intentions_vit:
        print(f"\n--- ViT Intention Prediction Results (on TP detections @ IoU>={IOU_THRESHOLD_FOR_INTENTION_MATCH}) ---")
        labels_report = list(range(NUM_INTENTION_CLASSES))
        target_names_report = [INTENTIONS_MAP_REV.get(i, f"Class_{i}") for i in labels_report]

        overall_acc_vit = accuracy_score(matched_gt_intentions_vit, matched_pred_intentions_vit)
        print(f"ViT Overall Accuracy: {overall_acc_vit:.4f}")

        f1_macro_vit = f1_score(matched_gt_intentions_vit, matched_pred_intentions_vit, labels=labels_report, average='macro', zero_division=0)
        f1_weighted_vit = f1_score(matched_gt_intentions_vit, matched_pred_intentions_vit, labels=labels_report, average='weighted', zero_division=0)
        print(f"ViT F1 (Macro):   {f1_macro_vit:.4f}")
        print(f"ViT F1 (Weighted): {f1_weighted_vit:.4f}")

        print("ViT F1 (Per Class):")
        f1_per_class_vit = f1_score(matched_gt_intentions_vit, matched_pred_intentions_vit, labels=labels_report, average=None, zero_division=0)
        for i, name in enumerate(target_names_report):
            print(f"  {name:<20}: {f1_per_class_vit[i]:.4f}")
    else:
        print(f"\nNo True Positive detections found for ViT model at IoU >= {IOU_THRESHOLD_FOR_INTENTION_MATCH} to evaluate intention.")

    print("\n--- Evaluation Script for ViT Finished ---")

if __name__ == '__main__':
    main_eval_vit()