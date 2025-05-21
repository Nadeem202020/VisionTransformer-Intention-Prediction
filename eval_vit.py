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
from torch.utils.data import DataLoader

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
    if hasattr(timm, '__version__'):
        print(f"timm version: {timm.__version__}")
    print(f"Using device: {DEVICE}")
    print(f"AV2 Map API Available: {AV2_MAP_AVAILABLE}")
    print(f"Shapely Available: {SHAPELY_AVAILABLE}")
    print(f"Evaluation using Rotated IoU for mAP/matching: {EVAL_USE_ROTATED_IOU} "
          f"(Requires Shapely if True)")
    if EVAL_USE_ROTATED_IOU and not SHAPELY_AVAILABLE:
        print("WARNING: EVAL_USE_ROTATED_IOU is True, but Shapely is not available. "
              "Falling back to axis-aligned IoU.")


    # ... (Validate Data Path - no change) ...
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
    # ... (no change in loading logic, ensure defaults are set for backbone_cfg) ...
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
        loaded_vit_backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
        loaded_vit_backbone_cfg.setdefault('lidar_input_channels', LIDAR_TOTAL_CHANNELS)
        loaded_vit_backbone_cfg.setdefault('map_input_channels', MAP_CHANNELS)
        loaded_vit_backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch8_224')
        loaded_vit_backbone_cfg.setdefault('vit_model_name_map', 'vit_tiny_patch8_224') # Original IntentNet uses tiny for map
        loaded_vit_backbone_cfg.setdefault('pretrained_lidar', False)
        loaded_vit_backbone_cfg.setdefault('pretrained_map', False)
        loaded_vit_backbone_cfg.setdefault('drop_path_rate_lidar', 0.1)
        loaded_vit_backbone_cfg.setdefault('drop_path_rate_map', 0.1)
        loaded_vit_backbone_cfg.setdefault('lidar_adapter_out_channels', 192)
        loaded_vit_backbone_cfg.setdefault('map_adapter_out_channels', 128) # Original IntentNet uses 128 for map ViT adapter

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
    # ... (no change in DataLoader logic) ...
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
    # Corrected anchor stride calculation
    vit_model_name_for_stride = loaded_vit_backbone_cfg.get('vit_model_name_lidar', 'vit_small_patch8_224')
    try:
        vit_patch_stride = int(vit_model_name_for_stride.split('_patch')[-1].split('_')[0])
    except ValueError:
        vit_patch_stride = 8
    fusion_block_stride_from_cfg = loaded_vit_backbone_cfg.get('fusion_block_stride', 1)
    actual_feature_map_stride_vit = vit_patch_stride * fusion_block_stride_from_cfg
    print(f"Using feature_map_stride = {actual_feature_map_stride_vit} for ViT anchor generation (ViT patch: {vit_patch_stride}, Fusion stride: {fusion_block_stride_from_cfg}).")

    try:
        anchors_for_vit_eval = generate_anchors(
            bev_height=GRID_HEIGHT_PX,
            bev_width=GRID_WIDTH_PX,
            feature_map_stride=actual_feature_map_stride_vit, # Use corrected stride
            anchor_configs=ANCHOR_CONFIGS_PAPER
        ).to(DEVICE)
        # Corrected print statement for anchor shape
        print(f"Anchors for ViT evaluation generated (stride {actual_feature_map_stride_vit}), shape: {anchors_for_vit_eval.shape}")
    except Exception as e_anchors:
        print(f"ERROR generating anchors for ViT: {e_anchors}")
        return
    print("--------------------")


    # --- Run Inference ---
    # ... (no change in inference loop) ...
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
                        cls_logits_sample = det_cls_logits[b_idx] # Should be [NumAnchors, 1]
                        box_preds_rel_sample = det_box_preds_rel[b_idx]
                        # int_logits_sample needs to be reshaped if not already [NumAnchors, NumClasses]
                        # In model_vit.py, intent_logits is reshaped to (B, -1, NUM_INTENTION_CLASSES)
                        # So, intent_logits[b_idx] is already [NumAnchors, NumClasses]
                        int_logits_sample = intent_logits[b_idx] 
                        scores_sample = torch.sigmoid(cls_logits_sample) # Sigmoid for objectness
                        
                        # Flatten scores if they are not already [NumAnchors]
                        if scores_sample.ndim > 1:
                            scores_sample = scores_sample.squeeze(-1)

                        keep_conf_indices = torch.where(scores_sample >= CONFIDENCE_THRESHOLD)[0]
                        if keep_conf_indices.numel() > 0:
                            scores_filtered = scores_sample[keep_conf_indices]
                            anchors_filtered = anchors_for_vit_eval[keep_conf_indices]
                            box_preds_rel_filtered = box_preds_rel_sample[keep_conf_indices]
                            int_logits_filtered = int_logits_sample[keep_conf_indices] # Already correct shape

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
        # Determine which IoU function to use
        iou_func_for_eval = compute_rotated_iou if (EVAL_USE_ROTATED_IOU and SHAPELY_AVAILABLE) else compute_axis_aligned_iou
        if EVAL_USE_ROTATED_IOU and not SHAPELY_AVAILABLE:
             iou_func_for_eval = compute_axis_aligned_iou # Fallback
        print(f"  Using IoU function for mAP: {iou_func_for_eval.__name__}")

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

                if iou_func_for_eval == compute_rotated_iou:
                    iou_matrix = iou_func_for_eval(pred_boxes_sorted.float(), gt_boxes.float())
                else:
                    iou_matrix = iou_func_for_eval(pred_boxes_sorted[:, :4].float(), gt_boxes[:, :4].float())
                
                gt_matched_flags = torch.zeros(num_gt, dtype=torch.bool, device=pred_boxes.device if pred_boxes.is_cuda else 'cpu')
                tp_flags = torch.zeros(num_pred, dtype=torch.bool, device=pred_boxes.device if pred_boxes.is_cuda else 'cpu')

                for pred_idx_sorted in range(num_pred):
                    pred_ious_with_gts = iou_matrix[pred_idx_sorted, :]
                    if pred_ious_with_gts.numel() == 0: continue
                    best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                    if best_iou_for_pred >= iou_thresh_eval:
                        if not gt_matched_flags[best_gt_idx_for_pred]:
                            tp_flags[pred_idx_sorted] = True
                            gt_matched_flags[best_gt_idx_for_pred] = True
                
                tp_cumsum = torch.cumsum(tp_flags.float(), dim=0)
                recall_steps = tp_cumsum / (num_gt + 1e-9)
                precision_steps = tp_cumsum / (torch.arange(1, num_pred + 1, device=tp_flags.device).float() + 1e-9)

                ap = calculate_ap(recall_steps.cpu().numpy(), precision_steps.cpu().numpy())
                average_precisions_per_iou_vit[iou_thresh_eval].append(ap)

    # ... (Rest of mAP printing - no change) ...
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
        iou_func_for_matching = compute_rotated_iou if (EVAL_USE_ROTATED_IOU and SHAPELY_AVAILABLE) else compute_axis_aligned_iou
        if EVAL_USE_ROTATED_IOU and not SHAPELY_AVAILABLE:
             iou_func_for_matching = compute_axis_aligned_iou # Fallback
        print(f"  Using IoU function for intention matching: {iou_func_for_matching.__name__}")

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

            if iou_func_for_matching == compute_rotated_iou:
                iou_matrix_intent = iou_func_for_matching(pred_boxes.float(), gt_boxes.float())
            else:
                iou_matrix_intent = iou_func_for_matching(pred_boxes[:, :4].float(), gt_boxes[:, :4].float())
            
            gt_matched_for_intent = torch.zeros(num_gt, dtype=torch.bool)
            sort_idx_intent = torch.argsort(pred_scores, descending=True)

            for i in range(num_pred_after_nms):
                pred_original_idx = sort_idx_intent[i]
                pred_ious_with_gts = iou_matrix_intent[pred_original_idx, :]
                if pred_ious_with_gts.numel() == 0: continue
                best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                if best_iou_for_pred >= IOU_THRESHOLD_FOR_INTENTION_MATCH:
                    if not gt_matched_for_intent[best_gt_idx_for_pred]:
                        gt_matched_for_intent[best_gt_idx_for_pred] = True
                        matched_pred_intentions_vit.append(pred_intentions[pred_original_idx].item())
                        matched_gt_intentions_vit.append(gt_intentions[best_gt_idx_for_pred].item())
    
    # ... (Rest of intention metrics printing - no change) ...
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