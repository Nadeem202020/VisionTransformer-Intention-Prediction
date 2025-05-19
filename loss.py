import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from constants import DOMINANT_CLASSES_FOR_DOWNSAMPLING, INTENTION_DOWNSAMPLE_RATIO
from utils import compute_axis_aligned_iou, compute_rotated_iou


class DetectionIntentionLoss(nn.Module):
    def __init__(self,
                 iou_threshold=0.6,
                 neg_iou_threshold=0.45,
                 box_weight=1.0,
                 cls_weight=1.0,
                 intent_weight=0.5,
                 intention_class_weights=None,
                 use_rotated_iou=False,
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 smooth_l1_beta=1.0 / 9.0,
                 apply_intention_downsampling=True,
                 dominant_intentions=DOMINANT_CLASSES_FOR_DOWNSAMPLING,
                 intention_downsample_ratio=INTENTION_DOWNSAMPLE_RATIO
                 ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.intent_weight = intent_weight
        self.use_rotated_iou = use_rotated_iou
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta

        self.apply_intention_downsampling = apply_intention_downsampling
        self.dominant_intentions = set(dominant_intentions)
        self.intention_downsample_keep_prob = 1.0 - intention_downsample_ratio

        effective_intention_weights = None
        if not self.apply_intention_downsampling and intention_class_weights is not None:
             effective_intention_weights = intention_class_weights

        self.register_buffer('final_intention_class_weights', effective_intention_weights)
        self.intention_criterion = nn.CrossEntropyLoss(weight=self.final_intention_class_weights, reduction='none')

        print(f"Loss Initialized: Use Rotated IoU: {self.use_rotated_iou}, Apply Intention Downsampling: {self.apply_intention_downsampling}, "
              f"Downsample Ratio: {intention_downsample_ratio if self.apply_intention_downsampling else 'N/A'}, Intent Weight: {self.intent_weight}")
        if self.final_intention_class_weights is not None:
            weights_to_print = self.final_intention_class_weights.cpu().numpy().round(4)
            print(f"  Using Intention Class Weights: {weights_to_print}")
        elif self.apply_intention_downsampling:
             print(f"  Using Intention Downsampling for classes: {self.dominant_intentions} (Keep Prob: {self.intention_downsample_keep_prob:.2f})")
        else:
             print("  Using standard CrossEntropyLoss for Intention (no weights/downsampling).")


    def forward(self, cls_logits, box_preds, intention_logits, anchors, gt_list):
        B = cls_logits.shape[0]
        N_total_anchors = anchors.shape[0]
        device = cls_logits.device
        anchors = anchors.to(device)

        cls_targets = torch.full((B, N_total_anchors), -1, dtype=torch.long, device=device)
        box_targets = torch.zeros((B, N_total_anchors, 6), dtype=torch.float32, device=device)
        intention_targets = torch.full((B, N_total_anchors), -1, dtype=torch.long, device=device)

        for b in range(B):
            if not isinstance(gt_list[b], dict) or 'boxes_xywha' not in gt_list[b] or 'intentions' not in gt_list[b]:
                cls_targets[b, :] = 0
                continue

            gt_boxes_item = gt_list[b]['boxes_xywha'].to(device)
            gt_intentions_item = gt_list[b]['intentions'].to(device)
            num_gt = gt_boxes_item.shape[0]

            if num_gt == 0:
                cls_targets[b, :] = 0
                continue

            iou_func = compute_rotated_iou if self.use_rotated_iou else compute_axis_aligned_iou
            current_anchors = anchors
            # Note: If using axis_aligned_iou, ensure current_anchors and gt_boxes_item are in xyxy or compatible format.
            # This logic might need to be adapted based on your specific IoU function inputs.
            # e.g., if anchors are [xc,yc,w,h,a] and AA IoU needs [x1,y1,x2,y2]:
            # if not self.use_rotated_iou:
            #     anchors_xyxy = convert_xywha_to_xyxy(current_anchors) # Implement this conversion
            #     gt_boxes_item_xyxy = convert_xywha_to_xyxy(gt_boxes_item) # Implement this
            #     iou_args = (anchors_xyxy, gt_boxes_item_xyxy)
            # else:
            iou_args = (current_anchors, gt_boxes_item)

            try:
                iou_matrix = iou_func(*iou_args)
            except Exception as e:
                print(f"Error in iou_func: {e}. Shapes: anchors={current_anchors.shape}, gt_boxes={gt_boxes_item.shape}")
                raise e

            max_iou_per_anchor, max_iou_gt_idx_per_anchor = iou_matrix.max(dim=1)

            neg_mask_item = max_iou_per_anchor < self.neg_iou_threshold
            cls_targets[b, neg_mask_item] = 0

            pos_mask_item = max_iou_per_anchor >= self.iou_threshold
            cls_targets[b, pos_mask_item] = 1
            # assigned_gt_indices_for_pos_anchors = max_iou_gt_idx_per_anchor[pos_mask_item] # Will be set later with final_pos_mask_item

            if num_gt > 0:
                 _, max_iou_anchor_idx_per_gt = iou_matrix.max(dim=0)
                 for gt_idx_force in range(num_gt):
                    anchor_idx_force = max_iou_anchor_idx_per_gt[gt_idx_force]
                    if not pos_mask_item[anchor_idx_force] and iou_matrix[anchor_idx_force, gt_idx_force] >= self.neg_iou_threshold:
                        pos_mask_item[anchor_idx_force] = True # Update local pos_mask_item
                        cls_targets[b, anchor_idx_force] = 1   # Update targets directly

            final_pos_mask_item = (cls_targets[b, :] == 1)
            assigned_gt_indices_for_pos_anchors = max_iou_gt_idx_per_anchor[final_pos_mask_item]

            pos_anchor_indices_item = torch.where(final_pos_mask_item)[0]

            if pos_anchor_indices_item.numel() > 0:
                current_assigned_anchors = anchors[pos_anchor_indices_item]
                current_assigned_gt_boxes = gt_boxes_item[assigned_gt_indices_for_pos_anchors]
                current_assigned_gt_intentions = gt_intentions_item[assigned_gt_indices_for_pos_anchors]

                eps = 1e-6
                delta_x = (current_assigned_gt_boxes[:, 0] - current_assigned_anchors[:, 0]) / (current_assigned_anchors[:, 2] + eps)
                delta_y = (current_assigned_gt_boxes[:, 1] - current_assigned_anchors[:, 1]) / (current_assigned_anchors[:, 3] + eps)
                delta_w = torch.log(current_assigned_gt_boxes[:, 2] / (current_assigned_anchors[:, 2] + eps) + eps)
                delta_l = torch.log(current_assigned_gt_boxes[:, 3] / (current_assigned_anchors[:, 3] + eps) + eps)
                delta_h_sin = torch.sin(current_assigned_gt_boxes[:, 4] - current_assigned_anchors[:, 4])
                delta_h_cos = torch.cos(current_assigned_gt_boxes[:, 4] - current_assigned_anchors[:, 4])

                box_targets[b, pos_anchor_indices_item, :] = torch.stack([delta_x, delta_y, delta_w, delta_l, delta_h_sin, delta_h_cos], dim=1)
                intention_targets[b, pos_anchor_indices_item] = current_assigned_gt_intentions

        cls_logits_flat = cls_logits.reshape(-1, 1)
        box_preds_flat = box_preds.reshape(-1, 6)
        intention_logits_flat = intention_logits.reshape(-1, intention_logits.shape[-1])

        cls_targets_flat = cls_targets.reshape(-1)
        box_targets_flat = box_targets.reshape(-1, 6)
        intention_targets_flat = intention_targets.reshape(-1)

        valid_cls_mask_flat = cls_targets_flat >= 0
        pos_targets_mask_flat = cls_targets_flat == 1
        num_pos_total_batch = pos_targets_mask_flat.sum()

        cls_loss = torch.tensor(0.0, device=device)
        if valid_cls_mask_flat.any():
            masked_cls_logits = cls_logits_flat[valid_cls_mask_flat]
            masked_cls_targets = cls_targets_flat[valid_cls_mask_flat].float()

            if masked_cls_logits.ndim > masked_cls_targets.ndim:
                masked_cls_targets = masked_cls_targets.unsqueeze(1)

            cls_loss = sigmoid_focal_loss(masked_cls_logits, masked_cls_targets,
                                         alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma,
                                         reduction="sum")
            cls_loss = cls_loss / max(1, num_pos_total_batch)

        box_loss = torch.tensor(0.0, device=device)
        if num_pos_total_batch > 0:
            masked_box_preds = box_preds_flat[pos_targets_mask_flat]
            masked_box_targets = box_targets_flat[pos_targets_mask_flat]
            box_loss = F.smooth_l1_loss(masked_box_preds, masked_box_targets,
                                        beta=self.smooth_l1_beta, reduction="sum")
            box_loss = box_loss / max(1, num_pos_total_batch)

        intent_loss = torch.tensor(0.0, device=device)
        if num_pos_total_batch > 0:
            intent_logits_pos = intention_logits_flat[pos_targets_mask_flat]
            intent_targets_pos = intention_targets_flat[pos_targets_mask_flat]

            if intent_targets_pos.numel() > 0:
                intent_loss_per_anchor = self.intention_criterion(intent_logits_pos, intent_targets_pos)

                if self.apply_intention_downsampling:
                    with torch.no_grad():
                        downsample_mask = torch.ones_like(intent_targets_pos, dtype=torch.float32)
                        for dominant_idx in self.dominant_intentions:
                            is_dominant_target_mask = (intent_targets_pos == dominant_idx)
                            if is_dominant_target_mask.any():
                                num_dominant_samples = is_dominant_target_mask.sum().item() # Ensure scalar for torch.rand
                                random_numbers_for_dominant = torch.rand(num_dominant_samples, device=device)
                                keep_mask_for_dominant = random_numbers_for_dominant < self.intention_downsample_keep_prob
                                downsample_mask[is_dominant_target_mask] = keep_mask_for_dominant.float()

                    intent_loss = (intent_loss_per_anchor * downsample_mask).sum()
                    effective_num_pos_intent = downsample_mask.sum()
                    intent_loss = intent_loss / max(1, effective_num_pos_intent)
                else:
                    intent_loss = intent_loss_per_anchor.sum() / max(1, intent_targets_pos.numel())

        total_loss = (self.cls_weight * cls_loss +
                      self.box_weight * box_loss +
                      self.intent_weight * intent_loss)

        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
             print(f"NaN or Inf DETECTED IN LOSS! Cls: {cls_loss.item()}, Box: {box_loss.item()}, Intent: {intent_loss.item()}")
             return {
                "loss": torch.tensor(0.0, device=device, requires_grad=True),
                "cls_loss": torch.tensor(0.0, device=device),
                "box_loss": torch.tensor(0.0, device=device),
                "intent_loss": torch.tensor(0.0, device=device),
                "num_pos_anchors": num_pos_total_batch.item() # Ensure scalar
            }

        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "box_loss": box_loss.detach(),
            "intent_loss": intent_loss.detach(),
            "num_pos_anchors": num_pos_total_batch.item() # Ensure scalar
        }