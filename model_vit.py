import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
import numpy as np

from constants import (LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX,
                       NUM_ANCHORS_PER_LOC, NUM_INTENTION_CLASSES)
from heads import DetectionHead, IntentionHead

def conv3x3_for_basic(in_planes: int, out_planes: int, stride: int = 1, kernel_size: int = 3) -> nn.Conv2d:
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def conv1x1_for_basic(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, kernel_size: int = 3):
        super().__init__()
        self.conv1 = conv3x3_for_basic(inplanes, planes, stride, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_for_basic(planes, planes, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; return self.relu(out)



class TwoStreamViTBackbone(nn.Module):
    def __init__(self,
                 lidar_input_channels: int = LIDAR_TOTAL_CHANNELS,
                 map_input_channels: int = MAP_CHANNELS,
                 vit_model_name_lidar: str = 'vit_small_patch8_224',
                 vit_model_name_map: str = 'vit_small_patch8_224',
                 pretrained_lidar: bool = False,
                 pretrained_map: bool = False,
                 img_size: tuple[int, int] = (GRID_HEIGHT_PX, GRID_WIDTH_PX),
                 drop_path_rate_lidar: float = 0.1,
                 drop_path_rate_map: float = 0.1,
                 lidar_adapter_out_channels: int = 192,
                 map_adapter_out_channels: int = 192,
                 fusion_block_planes: int = 512,
                 fusion_block_layers: int = 2,
                 fusion_block_kernel_size: int = 3,
                 fusion_block_stride: int = 1,
                 res_block_type: type[BasicBlock] = BasicBlock
                ):
        super().__init__()
        self.img_size = img_size
        self.lidar_adapter_out_channels = lidar_adapter_out_channels
        self.map_adapter_out_channels = map_adapter_out_channels

        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore", UserWarning)
            self.vit_lidar = timm.create_model(vit_model_name_lidar, pretrained=pretrained_lidar, in_chans=lidar_input_channels, img_size=self.img_size, drop_path_rate=drop_path_rate_lidar)
        self.vit_lidar.head = nn.Identity(); self.lidar_embed_dim = self.vit_lidar.embed_dim
        self.lidar_num_prefix_tokens = getattr(self.vit_lidar, 'num_prefix_tokens', 1 if hasattr(self.vit_lidar, 'cls_token') and self.vit_lidar.cls_token is not None else 0)
        self.lidar_grid_size, _ = self._get_patch_info(self.vit_lidar, "LiDAR")

        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore", UserWarning)
            self.vit_map = timm.create_model(vit_model_name_map, pretrained=pretrained_map, in_chans=map_input_channels, img_size=self.img_size, drop_path_rate=drop_path_rate_map)
        self.vit_map.head = nn.Identity(); self.map_embed_dim = self.vit_map.embed_dim
        self.map_num_prefix_tokens = getattr(self.vit_map, 'num_prefix_tokens', 1 if hasattr(self.vit_map, 'cls_token') and self.vit_map.cls_token is not None else 0)
        self.map_grid_size, _ = self._get_patch_info(self.vit_map, "Map")

        if self.lidar_grid_size != self.map_grid_size and self.lidar_grid_size is not None and self.map_grid_size is not None:
            warnings.warn(f"LiDAR patch grid {self.lidar_grid_size} and Map patch grid {self.map_grid_size} differ.")
        
        self.feature_map_grid_h = self.lidar_grid_size[0] if self.lidar_grid_size else (self.map_grid_size[0] if self.map_grid_size else 0)
        self.feature_map_grid_w = self.lidar_grid_size[1] if self.lidar_grid_size else (self.map_grid_size[1] if self.map_grid_size else 0)
        
        self.adapter_lidar = nn.Sequential(nn.LayerNorm(self.lidar_embed_dim), nn.Linear(self.lidar_embed_dim, self.lidar_adapter_out_channels), nn.GELU())
        self.adapter_map = nn.Sequential(nn.LayerNorm(self.map_embed_dim), nn.Linear(self.map_embed_dim, self.map_adapter_out_channels), nn.GELU())
            
        self.fusion_input_channels = self.lidar_adapter_out_channels + self.map_adapter_out_channels
        self.fusion_block_stride = fusion_block_stride 
        
        self.fusion_block = self._make_fusion_layer(res_block_type, fusion_block_planes, fusion_block_layers,
                                             stride=self.fusion_block_stride, 
                                             current_inplanes=self.fusion_input_channels,
                                             kernel_size_for_block=fusion_block_kernel_size)
        self.final_feature_channels = fusion_block_planes * res_block_type.expansion
        
        print(f"TwoStreamViTBackbone Initialized:")
        print(f"  LiDAR ViT: {vit_model_name_lidar}, Adapter Out: {self.lidar_adapter_out_channels}")
        print(f"  Map ViT: {vit_model_name_map}, Adapter Out: {self.map_adapter_out_channels}")
        print(f"  Input to Fusion Block: {self.fusion_input_channels} channels (after ViT adapters, at H/{self.lidar_grid_size[0] * self.img_size[0]//self.lidar_grid_size[0] if self.lidar_grid_size else 'Unknown'} x W/{self.lidar_grid_size[1] * self.img_size[1]//self.lidar_grid_size[1] if self.lidar_grid_size else 'Unknown'} resolution)")
        print(f"  Fusion Block: Stride {self.fusion_block_stride}, Output {self.final_feature_channels} channels")


    def _get_patch_info(self, vit_model: nn.Module, stream_name: str = "") -> tuple[tuple[int, int] | None, int]: 
        grid_size, num_patches = None, 0
        try:
            patch_embed = vit_model.patch_embed
            if hasattr(patch_embed, 'grid_size') and patch_embed.grid_size is not None: grid_size = tuple(patch_embed.grid_size); num_patches = grid_size[0] * grid_size[1]
            elif hasattr(patch_embed, 'num_patches'):
                num_patches = patch_embed.num_patches
                if self.img_size and hasattr(patch_embed, 'patch_size'):
                    patch_h, patch_w = patch_embed.patch_size if isinstance(patch_embed.patch_size, tuple) else (patch_embed.patch_size, patch_embed.patch_size)
                    gs_h, gs_w = self.img_size[0] // patch_h, self.img_size[1] // patch_w
                    if gs_h * gs_w == num_patches: grid_size = (gs_h, gs_w)
            if grid_size is None and num_patches > 0: print(f"Warning ({stream_name}): Could not reliably determine grid_size for {num_patches} patches.")
        except AttributeError: print(f"Error ({stream_name}): Could not access patch_embed attributes.")
        return grid_size, num_patches

    def _process_stream(self, x: torch.Tensor, vit_stream: nn.Module,
                        num_prefix_tokens: int, grid_size: tuple[int, int] | None,
                        adapter: nn.Module, stream_name: str) -> torch.Tensor | None: 
        tokens_all = vit_stream.forward_features(x); patch_tokens = tokens_all[:, num_prefix_tokens:]
        adapted_tokens = adapter(patch_tokens); B, N, C = adapted_tokens.shape
        if grid_size and N == grid_size[0] * grid_size[1]: Hf, Wf = grid_size; return adapted_tokens.permute(0, 2, 1).contiguous().view(B, C, Hf, Wf)
        else: print(f"ERROR ({stream_name}): Token count {N} or grid_size {grid_size} issue."); return None


    def _make_fusion_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int = 1, current_inplanes: int = 0, kernel_size_for_block: int = 3): # No change
        downsample = None; out_channels_block = planes * block.expansion
        if stride != 1 or current_inplanes != out_channels_block:
            downsample = nn.Sequential(conv1x1_for_basic(current_inplanes, out_channels_block, stride), nn.BatchNorm2d(out_channels_block))
        layers = [block(current_inplanes, planes, stride, downsample, kernel_size=kernel_size_for_block)]
        inplanes_for_rest_of_blocks = out_channels_block
        for _ in range(1, num_blocks): layers.append(block(inplanes_for_rest_of_blocks, planes, kernel_size=kernel_size_for_block))
        return nn.Sequential(*layers)

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> torch.Tensor: 
        lidar_feature_map = self._process_stream(lidar_bev, self.vit_lidar, self.lidar_num_prefix_tokens, self.lidar_grid_size, self.adapter_lidar, "LiDAR")
        if lidar_feature_map is None: return torch.zeros(lidar_bev.shape[0], self.final_feature_channels, self.feature_map_grid_h or 1, self.feature_map_grid_w or 1, device=lidar_bev.device)
        map_feature_map = self._process_stream(map_bev, self.vit_map, self.map_num_prefix_tokens, self.map_grid_size, self.adapter_map, "Map")
        if map_feature_map is None: return torch.zeros(map_bev.shape[0], self.final_feature_channels, self.feature_map_grid_h or 1, self.feature_map_grid_w or 1, device=map_bev.device)
        if lidar_feature_map.shape[2:] != map_feature_map.shape[2:]: map_feature_map = F.interpolate(map_feature_map, size=lidar_feature_map.shape[2:], mode='bilinear', align_corners=False)
        fused_vit_features = torch.cat([lidar_feature_map, map_feature_map], dim=1)
        final_features = self.fusion_block(fused_vit_features)
        return final_features


class IntentNetViT(nn.Module):
    def __init__(self, backbone_cfg: dict | None = None, head_cfg: dict | None = None):
        super().__init__()
        if backbone_cfg is None: backbone_cfg = {}
        backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch8_224')
        backbone_cfg.setdefault('vit_model_name_map', 'vit_small_patch8_224') 
        backbone_cfg.setdefault('pretrained_lidar', False)
        backbone_cfg.setdefault('pretrained_map', False)
        backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
        backbone_cfg.setdefault('lidar_adapter_out_channels', 192)
        backbone_cfg.setdefault('map_adapter_out_channels', 192) 
        backbone_cfg.setdefault('fusion_block_planes', 512)
        backbone_cfg.setdefault('fusion_block_layers', 2)
        backbone_cfg.setdefault('fusion_block_kernel_size', 3)
        backbone_cfg.setdefault('fusion_block_stride', 1) 

        self.backbone = TwoStreamViTBackbone(**backbone_cfg)
        feature_channels = self.backbone.final_feature_channels

        if head_cfg is None: head_cfg = {}
        self.det_head = DetectionHead(in_channels=feature_channels, **head_cfg)
        self.intention_head = IntentionHead(in_channels=feature_channels, num_classes=NUM_INTENTION_CLASSES, **head_cfg)
        
        print(f"IntentNetViT Heads Initialized. Input Channels: {feature_channels}")
        try:
            vit_patch_stride = int(backbone_cfg.get('vit_model_name_lidar', 'vit_small_patch8_224').split('_patch')[-1].split('_')[0])
        except ValueError:
            vit_patch_stride = 8 
            print(f"Warning: Could not parse patch stride from ViT name, defaulting to {vit_patch_stride}.")
            
        fusion_stride = backbone_cfg.get('fusion_block_stride', 1) 
        self.effective_head_stride = vit_patch_stride * fusion_stride
        print(f"  Effective total stride before heads: {self.effective_head_stride}x")

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # No change
        features = self.backbone(lidar_bev, map_bev)
        det_cls_logits, det_box_preds_rel = self.det_head(features); intention_logits = self.intention_head(features)
        B = features.shape[0]
        det_cls_logits = det_cls_logits.reshape(B, -1, 1); det_box_preds_rel = det_box_preds_rel.reshape(B, -1, 6)
        intention_logits = intention_logits.reshape(B, -1, NUM_INTENTION_CLASSES)
        return det_cls_logits, det_box_preds_rel, intention_logits