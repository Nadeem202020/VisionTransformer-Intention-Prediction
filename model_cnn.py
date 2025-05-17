import torch
import torch.nn as nn

from constants import LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, NUM_ANCHORS_PER_LOC, NUM_INTENTION_CLASSES
from heads import DetectionHead, IntentionHead

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class CNNBackbone(nn.Module):
    """Two-stream CNN backbone inspired by IntentNet for LiDAR and Map BEV feature extraction."""
    def __init__(self, block: type[BasicBlock] = BasicBlock,
                 lidar_input_channels: int = LIDAR_TOTAL_CHANNELS,
                 map_input_channels: int = MAP_CHANNELS,
                 lidar_init_features: int = 64, map_init_features: int = 32,
                 lidar_layers: list[int] = None, map_layers: list[int] = None): # Default to None, set in init
        super().__init__()
        self.block = block
        if lidar_layers is None: lidar_layers = [2, 2, 2]
        if map_layers is None: map_layers = [2, 2, 2]

        # LiDAR Stream
        self.lidar_inplanes = lidar_init_features
        self.lidar_conv1 = nn.Conv2d(lidar_input_channels, self.lidar_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.lidar_bn1 = nn.BatchNorm2d(self.lidar_inplanes)
        self.lidar_relu = nn.ReLU(inplace=True)
        self.lidar_layer1 = self._make_layer(block, lidar_init_features, lidar_layers[0], stride=1, prefix='lidar')
        lidar_planes_s2 = 160 // block.expansion
        self.lidar_layer2 = self._make_layer(block, lidar_planes_s2, lidar_layers[1], stride=2, prefix='lidar')
        lidar_planes_s3 = 224 // block.expansion
        self.lidar_layer3 = self._make_layer(block, lidar_planes_s3, lidar_layers[2], stride=2, prefix='lidar')
        self.lidar_output_channels = lidar_planes_s3 * block.expansion

        # Map Stream
        self.map_inplanes = map_init_features
        self.map_conv1 = nn.Conv2d(map_input_channels, self.map_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.map_bn1 = nn.BatchNorm2d(self.map_inplanes)
        self.map_relu = nn.ReLU(inplace=True)
        self.map_layer1 = self._make_layer(block, map_init_features, map_layers[0], stride=1, prefix='map')
        map_planes_s2 = 96 // block.expansion
        self.map_layer2 = self._make_layer(block, map_planes_s2, map_layers[1], stride=2, prefix='map')
        map_planes_s3 = 192 // block.expansion
        self.map_layer3 = self._make_layer(block, map_planes_s3, map_layers[2], stride=2, prefix='map')
        self.map_output_channels = map_planes_s3 * block.expansion

        self.final_feature_channels = self.lidar_output_channels + self.map_output_channels
        self._initialize_weights()
        # print(f"CNNBackbone Initialized: LiDAR Out={self.lidar_output_channels}, Map Out={self.map_output_channels}, Fusion={self.final_feature_channels}")

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int = 1, prefix: str = ''):
        downsample = None
        inplanes_attr_name = f"{prefix}_inplanes"
        current_inplanes = getattr(self, inplanes_attr_name)
        out_channels_block = planes * block.expansion

        if stride != 1 or current_inplanes != out_channels_block:
            downsample = nn.Sequential(
                conv1x1(current_inplanes, out_channels_block, stride),
                nn.BatchNorm2d(out_channels_block),
            )
        layers = [block(current_inplanes, planes, stride, downsample)]
        setattr(self, inplanes_attr_name, out_channels_block) # Update inplanes for the next _make_layer call
        
        # For subsequent blocks within this layer, inplanes is out_channels_block
        for _ in range(1, num_blocks):
            layers.append(block(out_channels_block, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> torch.Tensor:
        l_feat = self.lidar_relu(self.lidar_bn1(self.lidar_conv1(lidar_bev)))
        l_feat = self.lidar_layer1(l_feat)
        l_feat = self.lidar_layer2(l_feat)
        l_feat = self.lidar_layer3(l_feat)

        m_feat = self.map_relu(self.map_bn1(self.map_conv1(map_bev)))
        m_feat = self.map_layer1(m_feat)
        m_feat = self.map_layer2(m_feat)
        m_feat = self.map_layer3(m_feat)

        return torch.cat([l_feat, m_feat], dim=1)

class IntentNetCNN(nn.Module): # Renamed from IntentNetDetectorIntention
    """CNN-based model for joint detection and intention prediction."""
    def __init__(self, backbone_cfg: dict | None = None, head_cfg: dict | None = None):
        super().__init__()
        if backbone_cfg is None: backbone_cfg = {}
        self.backbone = CNNBackbone(**backbone_cfg)
        feature_channels = self.backbone.final_feature_channels

        if head_cfg is None: head_cfg = {}
        # num_anchors is handled by DetectionHead/IntentionHead defaults or passed in head_cfg
        # head_cfg.setdefault('num_anchors', NUM_ANCHORS_PER_LOC)

        self.det_head = DetectionHead(in_channels=feature_channels, **head_cfg)
        self.intention_head = IntentionHead(in_channels=feature_channels, num_classes=NUM_INTENTION_CLASSES, **head_cfg)
        # print(f"IntentNetCNN Heads Initialized. Input Channels: {feature_channels}")

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(lidar_bev, map_bev) # Expected shape [B, C_fused, H/8, W/8]
        det_cls_logits, det_box_preds_rel = self.det_head(features)
        intention_logits = self.intention_head(features)

        # Flatten outputs for the loss function
        B = features.shape[0]
        # det_cls_logits is [B, Hf, Wf, A] -> [B, Hf*Wf*A]
        det_cls_logits = det_cls_logits.reshape(B, -1)
        # det_box_preds_rel is [B, Hf, Wf, A, 6] -> [B, Hf*Wf*A, 6]
        det_box_preds_rel = det_box_preds_rel.reshape(B, -1, 6)
        # intention_logits is [B, Hf, Wf, A, NumClasses] -> [B, Hf*Wf*A, NumClasses]
        intention_logits = intention_logits.reshape(B, -1, NUM_INTENTION_CLASSES)
        
        return det_cls_logits, det_box_preds_rel, intention_logits