import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import torchvision 
from torch.utils.data import DataLoader


from constants import (GRID_HEIGHT_PX, GRID_WIDTH_PX, NUM_INTENTION_CLASSES,
                       ANCHOR_CONFIGS_PAPER, AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE,
                       INTENTIONS_MAP_REV, LIDAR_TOTAL_CHANNELS, MAP_CHANNELS)
from dataset import ArgoverseIntentNetDataset, collate_fn
from model_cnn import IntentNetCNN, BasicBlock
from utils import (generate_anchors, decode_box_predictions, apply_nms, compute_axis_aligned_iou, compute_rotated_iou, calculate_ap)


VAL_DATA_DIR = "path/to/your/argoverse2/sensor/val" 
MODEL_SAVE_PATH_CNN = "./trained_models_cnn/cnn_model.pth" 


CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.2
INFERENCE_BATCH_SIZE = 8 
NUM_WORKERS_EVAL = 0     
DETECTION_IOU_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9] 
IOU_THRESHOLD_FOR_INTENTION_MATCH = 0.5             
FEATURE_MAP_STRIDE_CNN = 8                          
EVAL_USE_ROTATED_IOU = False 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main_eval_cnn():
    """Main function to run evaluation for the CNN model."""
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Torch CUDA version: {torch.version.cuda}")
    print(f"AV2 Map API Available: {AV2_MAP_AVAILABLE}")
    print(f"Shapely Available: {SHAPELY_AVAILABLE}")
    print(f"Using device for CNN evaluation: {DEVICE}")
    print(f"Evaluation using Rotated IoU for mAP/matching: {EVAL_USE_ROTATED_IOU} "
          f"(Requires Shapely if True)")
    if EVAL_USE_ROTATED_IOU and not SHAPELY_AVAILABLE:
        print("WARNING: EVAL_USE_ROTATED_IOU is True, but Shapely is not available. "
              "Falling back to axis-aligned IoU.")

    val_data_path = Path(VAL_DATA_DIR)
    if not val_data_path.is_dir():
        print(f"ERROR: Evaluation data directory not found: {VAL_DATA_DIR}")
        print("Please update the VAL_DATA_DIR variable in this script.")
        return

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
        return

    loaded_cnn_backbone_cfg = checkpoint_cnn.get('backbone_cfg')
    if not loaded_cnn_backbone_cfg:
        print("ERROR: 'backbone_cfg' not found in the CNN checkpoint. Using default (may be incorrect).")
        loaded_cnn_backbone_cfg = {
            'block': BasicBlock,
            'lidar_input_channels': LIDAR_TOTAL_CHANNELS,
            'map_input_channels': MAP_CHANNELS
        }

    try:
        model_to_evaluate_cnn = IntentNetCNN(backbone_cfg=loaded_cnn_backbone_cfg).to(DEVICE)
        print("CNN Model structure instantiated successfully.")
    except Exception as e:
        print(f"ERROR: Failed to instantiate CNN model: {e}")
        return

    if 'model_state_dict' in checkpoint_cnn:
        try:
            model_to_evaluate_cnn.load_state_dict(checkpoint_cnn['model_state_dict'])
            print("TRAINED CNN Model state_dict loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load CNN model state_dict: {e}")
            return
    else:
        print("ERROR: 'model_state_dict' not found in the CNN checkpoint.")
        return

    model_to_evaluate_cnn.eval()
    print("TRAINED CNN Model set to evaluation mode.")
    print("--------------------")


    print("\n--- Preparing Evaluation DataLoader (for CNN) ---")
    try:
        eval_dataset_cnn = ArgoverseIntentNetDataset(data_dir=VAL_DATA_DIR, is_train=False)
        if not eval_dataset_cnn:
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
        return


    print("\n--- Running Inference with LOADED CNN Model ---")
    all_sample_results_cnn = []
    with torch.inference_mode():
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
                        
                        if scores_sample.ndim > 1:
                            scores_sample = scores_sample.squeeze(-1) 

                        keep_conf_indices = torch.where(scores_sample >= CONFIDENCE_THRESHOLD)[0]


                        if keep_conf_indices.numel() > 0:
                            scores_filtered = scores_sample[keep_conf_indices]
                            anchors_filtered = anchors_for_cnn_eval[keep_conf_indices]
                            box_preds_rel_filtered = box_preds_rel_sample[keep_conf_indices]
                            int_logits_filtered = int_logits[b_idx].reshape(-1, NUM_INTENTION_CLASSES)[keep_conf_indices]


                            boxes_decoded_abs = decode_box_predictions(box_preds_rel_filtered, anchors_filtered)
                            nms_keep_indices = apply_nms(boxes_decoded_abs, scores_filtered, NMS_IOU_THRESHOLD)

                            if nms_keep_indices.numel() > 0:
                                sample_preds['pred_scores'] = scores_filtered[nms_keep_indices].cpu()
                                sample_preds['pred_boxes_xywha'] = boxes_decoded_abs[nms_keep_indices].cpu()
                                sample_preds['pred_intentions'] = torch.argmax(int_logits_filtered[nms_keep_indices], dim=-1).cpu()
                    except Exception as e_post:
                        print(f"Error during post-processing CNN sample {b_idx} in batch {batch_idx}: {e_post}")

                    current_gt = gt_list_batch_cpu[b_idx]
                    all_sample_results_cnn.append({
                        **sample_preds,
                        'gt_boxes_xywha': current_gt.get('boxes_xywha', torch.empty((0,5), device='cpu')),
                        'gt_intentions': current_gt.get('intentions', torch.empty(0, dtype=torch.long, device='cpu'))
                    })
            except Exception as e_batch:
                print(f"!!! ERROR processing CNN eval batch {batch_idx}: {e_batch}")

    print(f"Collected results for {len(all_sample_results_cnn)} samples using TRAINED CNN model.")


    print("\n--- Calculating Detection mAP for TRAINED CNN Model ---")
    average_precisions_per_iou_cnn = {iou_thresh: [] for iou_thresh in DETECTION_IOU_THRESHOLDS}
    if not all_sample_results_cnn:
        print("WARNING: 'all_sample_results_cnn' is empty. Cannot calculate mAP.")
    else:
        iou_func_for_eval = compute_rotated_iou if (EVAL_USE_ROTATED_IOU and SHAPELY_AVAILABLE) else compute_axis_aligned_iou
        if EVAL_USE_ROTATED_IOU and not SHAPELY_AVAILABLE:
             iou_func_for_eval = compute_axis_aligned_iou 
        
        print(f"  Using IoU function for mAP: {iou_func_for_eval.__name__}")

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
                if num_gt == 0:
                    average_precisions_per_iou_cnn[iou_thresh_eval].append(0.0)
                    continue

                sort_idx = torch.argsort(pred_scores, descending=True)
                pred_boxes_sorted = pred_boxes[sort_idx]

                if iou_func_for_eval == compute_rotated_iou:
                    iou_matrix = iou_func_for_eval(pred_boxes_sorted.float(), gt_boxes.float())
                else: 
                    iou_matrix = iou_func_for_eval(pred_boxes_sorted[:, :4].float(), gt_boxes[:, :4].float())

                gt_matched_flags = torch.zeros(num_gt, dtype=torch.bool, device=pred_boxes.device if pred_boxes.is_cuda else 'cpu') 
                tp_flags = torch.zeros(num_pred, dtype=torch.bool, device=pred_boxes.device if pred_boxes.is_cuda else 'cpu')

                for pred_idx in range(num_pred):
                    pred_ious_with_gts = iou_matrix[pred_idx, :]
                    if pred_ious_with_gts.numel() == 0: continue
                    best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                    if best_iou_for_pred >= iou_thresh_eval:
                        if not gt_matched_flags[best_gt_idx_for_pred]:
                            tp_flags[pred_idx] = True
                            gt_matched_flags[best_gt_idx_for_pred] = True
                
                tp_cumsum = torch.cumsum(tp_flags.float(), dim=0)
                recall_steps = tp_cumsum / (num_gt + 1e-9)
                precision_steps = tp_cumsum / (torch.arange(1, num_pred + 1, device=tp_flags.device).float() + 1e-9)

                ap = calculate_ap(recall_steps.cpu().numpy(), precision_steps.cpu().numpy())
                average_precisions_per_iou_cnn[iou_thresh_eval].append(ap)

    print("\n--- TRAINED CNN Detection Results (mAP) ---")
    for iou_thresh_eval, ap_list in average_precisions_per_iou_cnn.items():
        mean_ap_cnn = np.mean(ap_list) if ap_list else 0.0
        print(f"TRAINED CNN mAP @ IoU={iou_thresh_eval:.1f}: {mean_ap_cnn:.4f}")


    print("\n--- Calculating Intention Prediction Metrics for TRAINED CNN Model ---")
    matched_pred_intentions_cnn = []
    matched_gt_intentions_cnn = []
    if not all_sample_results_cnn:
        print("WARNING: 'all_sample_results_cnn' is empty. Cannot calculate intention metrics.")
    else:
        iou_func_for_matching = compute_rotated_iou if (EVAL_USE_ROTATED_IOU and SHAPELY_AVAILABLE) else compute_axis_aligned_iou
        if EVAL_USE_ROTATED_IOU and not SHAPELY_AVAILABLE:
             iou_func_for_matching = compute_axis_aligned_iou 
        
        print(f"  Using IoU function for intention matching: {iou_func_for_matching.__name__}")

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
            
            if iou_func_for_matching == compute_rotated_iou:
                iou_matrix_intent = iou_func_for_matching(pred_boxes.float(), gt_boxes.float())
            else: 
                iou_matrix_intent = iou_func_for_matching(pred_boxes[:, :4].float(), gt_boxes[:, :4].float())

            gt_matched_for_intent = torch.zeros(num_gt, dtype=torch.bool)
            sort_idx_intent = torch.argsort(pred_scores, descending=True)

            for i in range(num_pred_after_nms):
                pred_idx = sort_idx_intent[i]
                pred_ious_with_gts = iou_matrix_intent[pred_idx, :]
                if pred_ious_with_gts.numel() == 0: continue
                best_iou_for_pred, best_gt_idx_for_pred = torch.max(pred_ious_with_gts, dim=0)

                if best_iou_for_pred >= IOU_THRESHOLD_FOR_INTENTION_MATCH:
                    if not gt_matched_for_intent[best_gt_idx_for_pred]:
                        gt_matched_for_intent[best_gt_idx_for_pred] = True
                        matched_pred_intentions_cnn.append(pred_intentions[pred_idx].item())
                        matched_gt_intentions_cnn.append(gt_intentions[best_gt_idx_for_pred].item())
    
    if matched_pred_intentions_cnn:
        print(f"\n--- TRAINED CNN Intention Prediction Results (on TP detections @ IoU>={IOU_THRESHOLD_FOR_INTENTION_MATCH}) ---")
        labels_report = list(range(NUM_INTENTION_CLASSES))
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