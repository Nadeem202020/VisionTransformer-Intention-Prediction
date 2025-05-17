import torch
import torch.nn as nn

# Assuming constants.py is in the same root directory
from constants import NUM_ANCHORS_PER_LOC, NUM_INTENTION_CLASSES

class DetectionHead(nn.Module):
    """
    Convolutional head for object detection.
    Outputs objectness scores and bounding box regression parameters for each anchor.
    """
    def __init__(self, in_channels: int, num_anchors: int = NUM_ANCHORS_PER_LOC):
        super().__init__()
        self.num_anchors = num_anchors
        num_box_params = 6  # cx, cy, w, l, sin(yaw_diff), cos(yaw_diff)
        num_outputs = self.num_anchors * (1 + num_box_params)
        self.conv = nn.Conv2d(in_channels, num_outputs, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.conv(x)
        B, _, Hf, Wf = out.shape
        num_box_params = 6
        out = out.view(B, self.num_anchors, (1 + num_box_params), Hf, Wf).permute(0, 3, 4, 1, 2).contiguous()
        cls_logits = out[..., 0]
        box_preds = out[..., 1:]
        return cls_logits, box_preds

class IntentionHead(nn.Module):
    """
    Convolutional head for intention prediction.
    Outputs logits for each intention class for each anchor.
    """
    def __init__(self, in_channels: int, num_anchors: int = NUM_ANCHORS_PER_LOC, num_classes: int = NUM_INTENTION_CLASSES):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        num_outputs = self.num_anchors * self.num_classes
        self.conv = nn.Conv2d(in_channels, num_outputs, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        B, _, Hf, Wf = out.shape
        out = out.view(B, self.num_anchors, self.num_classes, Hf, Wf).permute(0, 3, 4, 1, 2).contiguous()
        return out