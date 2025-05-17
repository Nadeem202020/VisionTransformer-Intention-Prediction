import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from constants import DOMINANT_CLASSES_FOR_DOWNSAMPLING, INTENTION_DOWNSAMPLE_RATIO
from utils import compute_axis_aligned_iou, compute_rotated_iou

class DetectionIntentionLoss(nn.Module):
    """
    Multi-task loss for object detection and intention prediction.
    - Classification: Focal Loss.
    - Bounding Box Regression: Smooth L1 Loss.
    - Intention Prediction: Cross-Entropy Loss (with optional downsampling).
    """
    def __init__(self,
                 iou_threshold: float = 0.6,
                 neg_iou_threshold: float = 0.45,
                 box_reg_weight: float = 1.0,
                 cls_focal_weight: float = 1.0,
                 intent_ce_weight: float = 0.5,
                 intention_class_weights: torch.Tensor | None = None, # For weighted CE if not downsampling
                 use_rotated_iou: bool = False,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 smooth_l1_beta: float = 1.0 / 9.0,
                 apply_intention_downsampling: bool = True,
                 dominant_intentions: set = DOMINANT_CLASSES_FOR_DOWNSAMPLING,
                 intention_downsample_ratio: float = INTENTION_DOWNSAMPLE_RATIO):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.box_reg_weight = box_reg_weight
        self.cls_focal_weight = cls_focal_weight
        self.intent_ce_weight = intent_ce_weight
        self.use_rotated_iou = use_rotated_iou
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta

        self.apply_intention_downsampling = apply_intention_downsampling
        self.dominant_intentions = dominant_intentions
        self.intention_downsample_keep_prob = 1.0 - intention_downsample_ratio

        effective_ce_weights = None
        if not self.apply_intention_downsampling and intention_class_weights is not None:
            effective_ce_weights = intention_class_weights
        
        self.register_buffer('ce_intention_weights', effective_ce_weights)
        self.intention_criterion = nn.CrossEntropyLoss(
            weight=self.ce_intention_weights, reduction='none'
        )
        # print(f"DetectionIntentionLoss Initialized. Downsampling: {self.apply_intention_downsampling}")

    def forward(self, cls_logits: torch.Tensor, box_preds_rel: torch.Tensor,
                intention_logits: torch.Tensor, anchors: torch.Tensor,
                gt_list: list[dict]) -> dict[str, torch.Tensor]:
        
        batch_size, num_total_anchors = cls_logits.shape[0], anchors.shape[0]
        device = cls_logits.device
        anchors_dev = anchors.to(device)

        cls_targets = torch.full((batch_size, num_total_anchors), -1, dtype=torch.long, device=device)
        box_targets_encoded = torch.zeros((batch_size, num_total_anchors, 6), dtype=torch.float32, device=device)
        intention_targets = torch.full((batch_size, num_total_anchors), -1, dtype=torch.long, device=device)

        for b_idx in range(batch_size):
            gt_boxes_item = gt_list[b_idx]['boxes_xywha'].to(device)
            gt_intentions_item = gt_list[b_idx]['intentions'].to(device)
            num_gt_item = gt_boxes_item.shape[0]

            if num_gt_item == 0:
                cls_targets[b_idx, :] = 0; continue

            iou_func = compute_rotated_iou if self.use_rotated_iou else compute_axis_aligned_iou
            iou_args = (anchors_dev, gt_boxes_item) if self.use_rotated_iou else \
                       (anchors_dev[:, :4], gt_boxes_item[:, :4])
            iou_matrix = iou_func(*iou_args)

            max_iou_per_anchor, max_iou_gt_idx_per_anchor = iou_matrix.max(dim=1)
            
            neg_mask_item = max_iou_per_anchor < self.neg_iou_threshold
            cls_targets[b_idx, neg_mask_item] = 0
            pos_mask_item = max_iou_per_anchor >= self.iou_threshold
            
            for gt_i in range(num_gt_item):
                gt_iou_with_anchors = iou_matrix[:, gt_i]
                best_anchor_iou_for_this_gt, best_anchor_idx_for_this_gt = gt_iou_with_anchors.max(dim=0)
                if best_anchor_iou_for_this_gt >= self.neg_iou_threshold:
                    pos_mask_item[best_anchor_idx_for_this_gt] = True
                    max_iou_gt_idx_per_anchor[best_anchor_idx_for_this_gt] = gt_i
            
            cls_targets[b_idx, pos_mask_item] = 1
            positive_anchor_indices_item = torch.where(pos_mask_item)[0]

            if positive_anchor_indices_item.numel() > 0:
                assigned_gt_indices = max_iou_gt_idx_per_anchor[positive_anchor_indices_item]
                pos_anchors_item = anchors_dev[positive_anchor_indices_item]
                matched_gt_boxes = gt_boxes_item[assigned_gt_indices]
                matched_gt_intentions = gt_intentions_item[assigned_gt_indices]

                eps = 1e-6
                delta_x = (matched_gt_boxes[:, 0] - pos_anchors_item[:, 0]) / (pos_anchors_item[:, 2] + eps)
                delta_y = (matched_gt_boxes[:, 1] - pos_anchors_item[:, 1]) / (pos_anchors_item[:, 3] + eps)
                delta_w = torch.log((matched_gt_boxes[:, 2] / (pos_anchors_item[:, 2] + eps)) + eps)
                delta_l = torch.log((matched_gt_boxes[:, 3] / (pos_anchors_item[:, 3] + eps)) + eps)
                heading_diff = matched_gt_boxes[:, 4] - pos_anchors_item[:, 4]
                delta_h_sin = torch.sin(heading_diff)
                delta_h_cos = torch.cos(heading_diff)
                
                box_targets_encoded[b_idx, positive_anchor_indices_item, :] = torch.stack(
                    [delta_x, delta_y, delta_w, delta_l, delta_h_sin, delta_h_cos], dim=1)
                intention_targets[b_idx, positive_anchor_indices_item] = matched_gt_intentions
        
        valid_cls_mask = cls_targets >= 0
        positive_sample_mask = cls_targets == 1
        num_pos = positive_sample_mask.sum().clamp(min=1.0) # Use float for division

        loss_cls = torch.tensor(0.0, device=device)
        if valid_cls_mask.any():
            loss_cls = sigmoid_focal_loss(cls_logits[valid_cls_mask], cls_targets[valid_cls_mask].float(),
                                     alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma,
                                     reduction="sum") / num_pos

        loss_box_reg = torch.tensor(0.0, device=device)
        if positive_sample_mask.any():
            loss_box_reg = F.smooth_l1_loss(box_preds_rel[positive_sample_mask],
                                        box_targets_encoded[positive_sample_mask],
                                        beta=self.smooth_l1_beta, reduction="sum") / num_pos
            
        loss_intent = torch.tensor(0.0, device=device)
        valid_intent_mask = intention_targets >= 0 # Should align with positive_sample_mask
        if valid_intent_mask.any():
            intent_logits_pos = intention_logits[valid_intent_mask]
            intent_targets_pos = intention_targets[valid_intent_mask]
            intent_loss_per_anchor = self.intention_criterion(intent_logits_pos, intent_targets_pos)

            if self.apply_intention_downsampling:
                with torch.no_grad():
                    downsample_mask = torch.ones_like(intent_targets_pos, dtype=torch.float32)
                    for dominant_idx in self.dominant_intentions:
                        is_dominant_mask = (intent_targets_pos == dominant_idx)
                        if is_dominant_mask.any():
                            rand_vals = torch.rand_like(intent_targets_pos[is_dominant_mask].float())
                            keep_mask = rand_vals < self.intention_downsample_keep_prob
                            downsample_mask[is_dominant_mask] = keep_mask.float()
                
                intent_loss_sum = (intent_loss_per_anchor * downsample_mask).sum()
                effective_num_pos_intent = downsample_mask.sum().clamp(min=1.0)
                loss_intent = intent_loss_sum / effective_num_pos_intent
            else:
                num_pos_intent = valid_intent_mask.sum().clamp(min=1.0)
                loss_intent = intent_loss_per_anchor.sum() / num_pos_intent
        
        total_loss = (self.cls_focal_weight * loss_cls +
                      self.box_reg_weight * loss_box_reg +
                      self.intent_ce_weight * loss_intent)

        if torch.isnan(total_loss):
            print(f"NaN Loss Warning: Cls={loss_cls.item():.4f}, Box={loss_box_reg.item():.4f}, Intent={loss_intent.item():.4f}")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True) # Prevent training crash

        return {
            "loss": total_loss,
            "cls_loss": loss_cls.detach(), "box_loss": loss_box_reg.detach(),
            "intent_loss": loss_intent.detach(), "num_pos_anchors": positive_sample_mask.sum()
        }