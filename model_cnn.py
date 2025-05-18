import torch
import torch.nn as nn

from constants import LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, NUM_ANCHORS_PER_LOC, NUM_INTENTION_CLASSES
from heads import DetectionHead, IntentionHead

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, kernel_size: int = 3):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size)
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
    """
    Two-stream CNN backbone that closely matches IntentNet paper's diagram (Fig 2c)
    """
    def __init__(self, block: type[BasicBlock] = BasicBlock,
                 lidar_input_channels: int = LIDAR_TOTAL_CHANNELS,
                 map_input_channels: int = MAP_CHANNELS,
                 # Channel counts from IntentNet paper
                 lidar_s1_planes: int = 160,
                 lidar_s2_planes: int = 192,
                 lidar_s3_planes: int = 224,
                 map_s1_planes: int = 32,
                 map_s2_planes: int = 64,
                 map_s3_planes: int = 96,
                 fusion_block_planes: int = 512,
                 fusion_block_layers: int = 2,
                 num_blocks_per_stage: int = 2,
                 res_block2_kernel_size: int = 5, # Paper uses k5 for Res_block_2 stages
                 fusion_block_kernel_size: int = 3 # Paper uses k3 for Res_block_3
                 ):
        super().__init__()
        self.block = block

        # LiDAR Stream
        # No initial conv. Input to stage1 is raw lidar_bev.
        current_lidar_inplanes = lidar_input_channels
        self.lidar_stage1 = self._make_layer(block, lidar_s1_planes, num_blocks_per_stage, stride=2, current_inplanes=current_lidar_inplanes, kernel_size_for_block=res_block2_kernel_size)
        current_lidar_inplanes = lidar_s1_planes * block.expansion
        self.lidar_stage2 = self._make_layer(block, lidar_s2_planes, num_blocks_per_stage, stride=1, current_inplanes=current_lidar_inplanes, kernel_size_for_block=res_block2_kernel_size)
        current_lidar_inplanes = lidar_s2_planes * block.expansion
        self.lidar_stage3 = self._make_layer(block, lidar_s3_planes, num_blocks_per_stage, stride=2, current_inplanes=current_lidar_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.lidar_output_channels = lidar_s3_planes * block.expansion

        # Map Stream
        # No initial conv. Input to stage1 is raw map_bev.
        current_map_inplanes = map_input_channels
        self.map_stage1 = self._make_layer(block, map_s1_planes, num_blocks_per_stage, stride=2, current_inplanes=current_map_inplanes, kernel_size_for_block=res_block2_kernel_size)
        current_map_inplanes = map_s1_planes * block.expansion
        self.map_stage2 = self._make_layer(block, map_s2_planes, num_blocks_per_stage, stride=1, current_inplanes=current_map_inplanes, kernel_size_for_block=res_block2_kernel_size)
        current_map_inplanes = map_s2_planes * block.expansion
        self.map_stage3 = self._make_layer(block, map_s3_planes, num_blocks_per_stage, stride=2, current_inplanes=current_map_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.map_output_channels = map_s3_planes * block.expansion

        # Fusion Subnetwork (Res_block_3 in paper)
        self.fusion_inplanes = self.lidar_output_channels + self.map_output_channels
        # This block downsamples by /2 and outputs fusion_block_planes channels
        self.fusion_block = self._make_layer(block, fusion_block_planes, fusion_block_layers, stride=2, current_inplanes=self.fusion_inplanes, kernel_size_for_block=fusion_block_kernel_size)
        self.final_feature_channels = fusion_block_planes * block.expansion

        self._initialize_weights()
        print(f"CNNBackbone (No Initial Conv, Paper Fig2c Downsampling) Initialized:")
        print(f"  LiDAR Input Channels: {lidar_input_channels}, Output (pre-fusion): {self.lidar_output_channels}")
        print(f"  Map Input Channels: {map_input_channels}, Output (pre-fusion): {self.map_output_channels}")
        print(f"  Input to Fusion Block: {self.fusion_inplanes}, Final Output: {self.final_feature_channels}")
        print(f"  Stream pre-fusion spatial downsampling: 4x")
        print(f"  Final feature spatial downsampling: 8x")


    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int = 1, current_inplanes: int = 0, kernel_size_for_block: int = 3):
        downsample = None
        out_channels_block = planes * block.expansion

        if stride != 1 or current_inplanes != out_channels_block:
            downsample = nn.Sequential(
                conv1x1(current_inplanes, out_channels_block, stride),
                nn.BatchNorm2d(out_channels_block),
            )
        layers = [block(current_inplanes, planes, stride, downsample, kernel_size=kernel_size_for_block)]
        
        inplanes_for_rest_of_blocks = out_channels_block
        for _ in range(1, num_blocks):
            layers.append(block(inplanes_for_rest_of_blocks, planes, kernel_size=kernel_size_for_block))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> torch.Tensor:
        l_feat = self.lidar_stage1(lidar_bev)   # Input: B x LidarCh x H_in x W_in
        l_feat = self.lidar_stage2(l_feat)
        l_feat = self.lidar_stage3(l_feat)      # Output: B x 224 x H_in/4 x W_in/4

        m_feat = self.map_stage1(map_bev)       # Input: B x MapCh x H_in x W_in
        m_feat = self.map_stage2(m_feat)
        m_feat = self.map_stage3(m_feat)        # Output: B x 96 x H_in/4 x W_in/4

        fused_feat_pre_block3 = torch.cat([l_feat, m_feat], dim=1) # Output: B x 320 x H_in/4 x W_in/4
        
        # Pass through the shared fusion block (Res_block_3 equivalent)
        final_features = self.fusion_block(fused_feat_pre_block3)  # Output: B x 512 x H_in/8 x W_in/8
        
        return final_features

class IntentNetCNN(nn.Module):
    """CNN-based model for joint detection and intention prediction,
       using the backbone that matches IntentNet paper (Fig 2c downsampling)."""
    def __init__(self, backbone_cfg: dict | None = None, head_cfg: dict | None = None):
        super().__init__()
        if backbone_cfg is None: backbone_cfg = {}
        self.backbone = CNNBackbone(**backbone_cfg)
        feature_channels = self.backbone.final_feature_channels

        if head_cfg is None: head_cfg = {}
        self.det_head = DetectionHead(in_channels=feature_channels, **head_cfg)
        self.intention_head = IntentionHead(in_channels=feature_channels, num_classes=NUM_INTENTION_CLASSES, **head_cfg)
        print(f"IntentNetCNN (Paper Fig2c Aligned Downsampling) Heads Initialized. Input Channels: {feature_channels}")

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: B x 512 x H_in/8 x W_in/8
        features = self.backbone(lidar_bev, map_bev)
        
        det_cls_logits, det_box_preds_rel = self.det_head(features)
        intention_logits = self.intention_head(features)

        # Flatten outputs for the loss function.
        # Assumes heads output (B, H_feat, W_feat, Num_Anchors_x_Values)
        # and need to be reshaped to (B, NumTotalAnchors, Values_per_Anchor).
        B = features.shape[0]
        
        det_cls_logits = det_cls_logits.reshape(B, -1, 1) 
        det_box_preds_rel = det_box_preds_rel.reshape(B, -1, 6)
        intention_logits = intention_logits.reshape(B, -1, NUM_INTENTION_CLASSES)
        
        return det_cls_logits, det_box_preds_rel, intention_logits