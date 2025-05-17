import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import warnings
import numpy as np

from constants import (LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX,
                       NUM_ANCHORS_PER_LOC, NUM_INTENTION_CLASSES)
from heads import DetectionHead, IntentionHead

class TwoStreamViTBackbone(nn.Module):
    """
    Two-stream Vision Transformer backbone.
    One ViT processes LiDAR BEV, another processes Map BEV.
    Features are adapted, reshaped to 2D, and concatenated.
    """
    def __init__(self,
                 lidar_input_channels: int = LIDAR_TOTAL_CHANNELS,
                 map_input_channels: int = MAP_CHANNELS,
                 vit_model_name_lidar: str = 'vit_small_patch16_224',
                 vit_model_name_map: str = 'vit_tiny_patch16_224',
                 pretrained_lidar: bool = False, # Defaulting to False as per training
                 pretrained_map: bool = False,   # Defaulting to False as per training
                 img_size: tuple[int, int] = (GRID_HEIGHT_PX, GRID_WIDTH_PX),
                 drop_path_rate_lidar: float = 0.1,
                 drop_path_rate_map: float = 0.1,
                 lidar_adapter_out_channels: int = 192, # Defaulting to your training config
                 map_adapter_out_channels: int = 128     # Defaulting to your training config
                ):
        super().__init__()
        self.img_size = img_size
        self.lidar_adapter_out_channels = lidar_adapter_out_channels
        self.map_adapter_out_channels = map_adapter_out_channels

        # LiDAR Stream ViT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.vit_lidar = timm.create_model(
                vit_model_name_lidar, pretrained=pretrained_lidar,
                in_chans=lidar_input_channels, img_size=self.img_size,
                drop_path_rate=drop_path_rate_lidar
            )
        self.vit_lidar.head = nn.Identity()
        self.lidar_embed_dim = self.vit_lidar.embed_dim
        self.lidar_num_prefix_tokens = getattr(self.vit_lidar, 'num_prefix_tokens',
                                              1 if hasattr(self.vit_lidar, 'cls_token') and self.vit_lidar.cls_token is not None else 0)
        self.lidar_grid_size, _ = self._get_patch_info(self.vit_lidar, "LiDAR")

        # Map Stream ViT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.vit_map = timm.create_model(
                vit_model_name_map, pretrained=pretrained_map,
                in_chans=map_input_channels, img_size=self.img_size,
                drop_path_rate=drop_path_rate_map
            )
        self.vit_map.head = nn.Identity()
        self.map_embed_dim = self.vit_map.embed_dim
        self.map_num_prefix_tokens = getattr(self.vit_map, 'num_prefix_tokens',
                                            1 if hasattr(self.vit_map, 'cls_token') and self.vit_map.cls_token is not None else 0)
        self.map_grid_size, _ = self._get_patch_info(self.vit_map, "Map")

        if self.lidar_grid_size != self.map_grid_size and \
           self.lidar_grid_size is not None and self.map_grid_size is not None:
            warnings.warn(f"LiDAR patch grid {self.lidar_grid_size} and Map patch grid {self.map_grid_size} differ. "
                          "Interpolation might be needed if spatial dimensions don't match after reshaping.")
        
        self.feature_map_grid_h = self.lidar_grid_size[0] if self.lidar_grid_size else (self.map_grid_size[0] if self.map_grid_size else 0)
        self.feature_map_grid_w = self.lidar_grid_size[1] if self.lidar_grid_size else (self.map_grid_size[1] if self.map_grid_size else 0)
        
        # Adapter Layers
        self.adapter_lidar = nn.Sequential(
            nn.LayerNorm(self.lidar_embed_dim),
            nn.Linear(self.lidar_embed_dim, self.lidar_adapter_out_channels),
            nn.GELU())
        self.adapter_map = nn.Sequential(
            nn.LayerNorm(self.map_embed_dim),
            nn.Linear(self.map_embed_dim, self.map_adapter_out_channels),
            nn.GELU())
            
        self.final_feature_channels = self.lidar_adapter_out_channels + self.map_adapter_out_channels
        # print(f"TwoStreamViTBackbone Initialized. Fused Channels: {self.final_feature_channels}")

    def _get_patch_info(self, vit_model: nn.Module, stream_name: str = "") -> tuple[tuple[int, int] | None, int]:
        grid_size, num_patches = None, 0
        try:
            patch_embed = vit_model.patch_embed
            if hasattr(patch_embed, 'grid_size') and patch_embed.grid_size is not None:
                grid_size = tuple(patch_embed.grid_size)
                num_patches = grid_size[0] * grid_size[1]
            elif hasattr(patch_embed, 'num_patches'):
                num_patches = patch_embed.num_patches
                if self.img_size and hasattr(patch_embed, 'patch_size'): # Try to infer
                    patch_h, patch_w = patch_embed.patch_size if isinstance(patch_embed.patch_size, tuple) else (patch_embed.patch_size, patch_embed.patch_size)
                    gs_h, gs_w = self.img_size[0] // patch_h, self.img_size[1] // patch_w
                    if gs_h * gs_w == num_patches: grid_size = (gs_h, gs_w)
            if grid_size is None and num_patches > 0:
                print(f"Warning ({stream_name}): Could not reliably determine grid_size for {num_patches} patches.")
        except AttributeError:
            print(f"Error ({stream_name}): Could not access patch_embed attributes.")
        return grid_size, num_patches

    def _process_stream(self, x: torch.Tensor, vit_stream: nn.Module,
                        num_prefix_tokens: int, grid_size: tuple[int, int] | None,
                        adapter: nn.Module, stream_name: str) -> torch.Tensor | None:
        tokens_all = vit_stream.forward_features(x)
        patch_tokens = tokens_all[:, num_prefix_tokens:]
        adapted_tokens = adapter(patch_tokens)
        
        B, N, C = adapted_tokens.shape
        if grid_size and N == grid_size[0] * grid_size[1]:
            Hf, Wf = grid_size
            return adapted_tokens.permute(0, 2, 1).contiguous().view(B, C, Hf, Wf)
        else:
            print(f"ERROR ({stream_name}): Token count {N} or grid_size {grid_size} issue. Cannot reshape.")
            return None

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> torch.Tensor:
        lidar_feature_map = self._process_stream(lidar_bev, self.vit_lidar, self.lidar_num_prefix_tokens,
                                                 self.lidar_grid_size, self.adapter_lidar, "LiDAR")
        if lidar_feature_map is None:
            return torch.zeros(lidar_bev.shape[0], self.final_feature_channels,
                               self.feature_map_grid_h or 1, self.feature_map_grid_w or 1, # Fallback size
                               device=lidar_bev.device)

        map_feature_map = self._process_stream(map_bev, self.vit_map, self.map_num_prefix_tokens,
                                               self.map_grid_size, self.adapter_map, "Map")
        if map_feature_map is None:
            return torch.zeros(map_bev.shape[0], self.final_feature_channels,
                               self.feature_map_grid_h or 1, self.feature_map_grid_w or 1,
                               device=map_bev.device)

        if lidar_feature_map.shape[2:] != map_feature_map.shape[2:]:
            # This condition should ideally be handled by ensuring ViTs chosen or img_size/patch_size result in same grid
            # print(f"Warning: LiDAR features {lidar_feature_map.shape[2:]} and Map features {map_feature_map.shape[2:]} differ. Upsampling map.")
            map_feature_map = F.interpolate(map_feature_map, size=lidar_feature_map.shape[2:],
                                            mode='bilinear', align_corners=False)
        
        return torch.cat([lidar_feature_map, map_feature_map], dim=1)

class IntentNetViT(nn.Module): # Renamed from IntentNetDetectorIntentionTwoStreamViT
    """ViT-based model for joint detection and intention prediction."""
    def __init__(self, backbone_cfg: dict | None = None, head_cfg: dict | None = None):
        super().__init__()
        if backbone_cfg is None: backbone_cfg = {}
        # Ensure defaults from training script are reflected here if not passed
        backbone_cfg.setdefault('vit_model_name_lidar', 'vit_small_patch16_224')
        backbone_cfg.setdefault('vit_model_name_map', 'vit_tiny_patch16_224')
        backbone_cfg.setdefault('pretrained_lidar', False)
        backbone_cfg.setdefault('pretrained_map', False)
        backbone_cfg.setdefault('img_size', (GRID_HEIGHT_PX, GRID_WIDTH_PX))
        backbone_cfg.setdefault('lidar_adapter_out_channels', 192)
        backbone_cfg.setdefault('map_adapter_out_channels', 128)

        self.backbone = TwoStreamViTBackbone(**backbone_cfg)
        feature_channels = self.backbone.final_feature_channels

        if head_cfg is None: head_cfg = {}
        # head_cfg.setdefault('num_anchors', NUM_ANCHORS_PER_LOC) # Defaults in Head classes

        self.det_head = DetectionHead(in_channels=feature_channels, **head_cfg)
        self.intention_head = IntentionHead(in_channels=feature_channels, num_classes=NUM_INTENTION_CLASSES, **head_cfg)
        # print(f"IntentNetViT Heads Initialized. Input Channels: {feature_channels}")

    def forward(self, lidar_bev: torch.Tensor, map_bev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(lidar_bev, map_bev)
        det_cls_logits, det_box_preds_rel = self.det_head(features)
        intention_logits = self.intention_head(features)

        B = features.shape[0]
        det_cls_logits = det_cls_logits.reshape(B, -1)
        det_box_preds_rel = det_box_preds_rel.reshape(B, -1, 6)
        intention_logits = intention_logits.reshape(B, -1, NUM_INTENTION_CLASSES)
        
        return det_cls_logits, det_box_preds_rel, intention_logits