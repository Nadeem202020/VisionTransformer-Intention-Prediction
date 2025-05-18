import torch
import torch.nn as nn

from constants import LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, NUM_ANCHORS_PER_LOC, NUM_INTENTION_CLASSES
from heads import DetectionHead, IntentionHead

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, kernel_size: int = 3) -> nn.Conv2d:
    padding = (kernel_size -1) // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

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
    Two-stream CNN backbone modified to more closely match IntentNet paper's diagram (Fig 2c),
    including the post-fusion Res_block_3.
    Assumes BasicBlock uses 3x3 kernels internally unless specified.
    The paper hints at 'k5' for Res_block_2 stages, but for simplicity and standard
    ResNet structure, we'll use 3x3 kernels within BasicBlock here.
    The channel counts and downsampling are prioritized to match the diagram.
    """
    def __init__(self, block: type[BasicBlock] = BasicBlock,
                 lidar_input_channels: int = LIDAR_TOTAL_CHANNELS,
                 map_input_channels: int = MAP_CHANNELS,
                 # --- LiDAR Stream Params (to match diagram after initial conv) ---
                 lidar_init_features: int = 64, # Channels after initial 7x7 conv
                 lidar_s1_planes: int = 160,    # Corresponds to f160 after first Res_block_2 stage
                 lidar_s2_planes: int = 192,    # Corresponds to f192 after second Res_block_2 stage
                 lidar_s3_planes: int = 224,    # Corresponds to f224 after third Res_block_2 stage (output of LiDAR stream)
                 # --- Map Stream Params (to match diagram after initial conv) ---
                 map_init_features: int = 32,   # Channels after initial 7x7 conv
                 map_s1_planes: int = 32,       # Corresponds to f32 after first Res_block_2 stage
                 map_s2_planes: int = 64,       # Corresponds to f64 after second Res_block_2 stage
                 map_s3_planes: int = 96,       # Corresponds to f96 after third Res_block_2 stage (output of Map stream)
                 # --- Shared Fusion Block Params ---
                 fusion_block_planes: int = 512, # Corresponds to f512 of Res_block_3
                 fusion_block_layers: int = 2,   # Number of blocks in Res_block_3 (e.g., 2 for a ResNet18/34 style layer)
                 # --- General Layer Config (number of blocks per stage) ---
                 # Assuming 2 blocks per stage like a ResNet-18 style for each Res_block_2 stage in diagram
                 # And for the Res_block_3 stage
                 num_blocks_per_stage: int = 2,
                 res_block2_kernel_size: int = 5, # For LiDAR and Map streams' "Res_block_2"
                 fusion_block_kernel_size: int = 3 # For the post-fusion "Res_block_3"
                 ):
        super().__init__()
        self.block = block

        # --- LiDAR Stream (Matches paper's f160, f192, f224 sequence) ---
        self.lidar_inplanes = lidar_init_features # After initial 7x7 conv
        self.lidar_conv1 = nn.Conv2d(lidar_input_channels, self.lidar_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.lidar_bn1 = nn.BatchNorm2d(self.lidar_inplanes)
        self.lidar_relu1 = nn.ReLU(inplace=True) # Renamed for clarity

        # Stage 1 (e.g., -> f160, downsamples /2)
        self.lidar_stage1 = self._make_layer(block, lidar_s1_planes, num_blocks_per_stage, stride=2, prefix='lidar', current_inplanes=self.lidar_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.lidar_inplanes = lidar_s1_planes * block.expansion # Update inplanes for next stage
        # Stage 2 (e.g., -> f192, stride 1)
        self.lidar_stage2 = self._make_layer(block, lidar_s2_planes, num_blocks_per_stage, stride=1, prefix='lidar', current_inplanes=self.lidar_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.lidar_inplanes = lidar_s2_planes * block.expansion
        # Stage 3 (e.g., -> f224, downsamples /2) - This is the output for LiDAR stream to concat
        self.lidar_stage3 = self._make_layer(block, lidar_s3_planes, num_blocks_per_stage, stride=2, prefix='lidar', current_inplanes=self.lidar_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.lidar_output_channels = lidar_s3_planes * block.expansion # Should be 224

        # --- Map Stream (Matches paper's f32, f64, f96 sequence) ---
        self.map_inplanes = map_init_features # After initial 7x7 conv
        self.map_conv1 = nn.Conv2d(map_input_channels, self.map_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.map_bn1 = nn.BatchNorm2d(self.map_inplanes)
        self.map_relu1 = nn.ReLU(inplace=True) # Renamed for clarity

        # Stage 1 (e.g., -> f32, downsamples /2)
        self.map_stage1 = self._make_layer(block, map_s1_planes, num_blocks_per_stage, stride=2, prefix='map', current_inplanes=self.map_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.map_inplanes = map_s1_planes * block.expansion
        # Stage 2 (e.g., -> f64, stride 1)
        self.map_stage2 = self._make_layer(block, map_s2_planes, num_blocks_per_stage, stride=1, prefix='map', current_inplanes=self.map_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.map_inplanes = map_s2_planes * block.expansion
        # Stage 3 (e.g., -> f96, downsamples /2) - This is the output for Map stream to concat
        self.map_stage3 = self._make_layer(block, map_s3_planes, num_blocks_per_stage, stride=2, prefix='map', current_inplanes=self.map_inplanes, kernel_size_for_block=res_block2_kernel_size)
        self.map_output_channels = map_s3_planes * block.expansion # Should be 96

        # --- Fusion Subnetwork (Res_block_3 in paper) ---
        # Input channels to this block will be lidar_output_channels + map_output_channels
        self.fusion_inplanes = self.lidar_output_channels + self.map_output_channels # 224 + 96 = 320
        # This block downsamples by /2 and outputs f512
        self.fusion_block = self._make_layer(block, fusion_block_planes, fusion_block_layers, stride=2, prefix='fusion', current_inplanes=self.fusion_inplanes, kernel_size_for_block=fusion_block_kernel_size)
        self.final_feature_channels = fusion_block_planes * block.expansion # Should be 512

        self._initialize_weights()
        print(f"CNNBackboneIntentNetPaper Initialized:")
        print(f"  LiDAR Stream Output (pre-fusion): {self.lidar_output_channels} channels")
        print(f"  Map Stream Output (pre-fusion): {self.map_output_channels} channels")
        print(f"  Input to Fusion Block: {self.fusion_inplanes} channels")
        print(f"  Final Output (post-fusion block): {self.final_feature_channels} channels")


    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int = 1, prefix: str = '', current_inplanes: int = 0, kernel_size_for_block: int = 3):
        # 'current_inplanes' is passed explicitly to avoid relying on class attributes that might change order
        downsample = None
        out_channels_block = planes * block.expansion

        if stride != 1 or current_inplanes != out_channels_block:
            downsample = nn.Sequential(
                conv1x1(current_inplanes, out_channels_block, stride),
                nn.BatchNorm2d(out_channels_block),
            )
        layers = [block(current_inplanes, planes, stride, downsample, kernel_size=kernel_size_for_block)] # First block handles downsampling
        
        # For subsequent blocks within this layer, inplanes is out_channels_block
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
        # LiDAR Stream
        l_feat = self.lidar_relu1(self.lidar_bn1(self.lidar_conv1(lidar_bev))) # Initial conv
        l_feat = self.lidar_stage1(l_feat)
        l_feat = self.lidar_stage2(l_feat)
        l_feat = self.lidar_stage3(l_feat) # Output: B x 224 x H/8 x W/8

        # Map Stream
        m_feat = self.map_relu1(self.map_bn1(self.map_conv1(map_bev))) # Initial conv
        m_feat = self.map_stage1(m_feat)
        m_feat = self.map_stage2(m_feat)
        m_feat = self.map_stage3(m_feat) # Output: B x 96 x H/8 x W/8

        # Concatenate features from two streams
        fused_feat_pre_block3 = torch.cat([l_feat, m_feat], dim=1) # Output: B x 320 x H/8 x W/8
        
        # Pass through the shared fusion block (Res_block_3 equivalent)
        final_features = self.fusion_block(fused_feat_pre_block3) # Output: B x 512 x H/16 x W/16
        
        return final_features

class IntentNetCNN(nn.Module):
    """CNN-based model for joint detection and intention prediction,
       using the backbone modified to match IntentNet paper."""
    def __init__(self, backbone_cfg: dict | None = None, head_cfg: dict | None = None):
        super().__init__()
        if backbone_cfg is None: backbone_cfg = {} # Allow passing custom block counts etc.
        # Ensure the new backbone is used
        self.backbone = CNNBackbone(**backbone_cfg)
        feature_channels = self.backbone.final_feature_channels # Should now be 512

        if head_cfg is None: head_cfg = {}
        self.det_head = DetectionHead(in_channels=feature_channels, **head_cfg)
        self.intention_head = IntentionHead(in_channels=feature_channels, num_classes=NUM_INTENTION_CLASSES, **head_cfg)
        print(f"IntentNetCNN (Paper Aligned) Heads Initialized. Input Channels: {feature_channels}")

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features will now be B x 512 x H/16 x W/16
        features = self.backbone(lidar_bev, map_bev)
        
        det_cls_logits, det_box_preds_rel = self.det_head(features)
        intention_logits = self.intention_head(features)

        # Flatten outputs for the loss function
        B = features.shape[0]
        det_cls_logits = det_cls_logits.reshape(B, -1)
        det_box_preds_rel = det_box_preds_rel.reshape(B, -1, 6)
        intention_logits = intention_logits.reshape(B, -1, NUM_INTENTION_CLASSES)
        
        return det_cls_logits, det_box_preds_rel, intention_logits