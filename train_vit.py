import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
# import timm # Not directly used in this script

# Project-specific imports
from constants import (GRID_HEIGHT_PX, GRID_WIDTH_PX, NUM_INTENTION_CLASSES, ANCHOR_CONFIGS_PAPER,
                       AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE,
                       DOMINANT_CLASSES_FOR_DOWNSAMPLING, INTENTION_DOWNSAMPLE_RATIO)
from dataset import ArgoverseIntentNetDataset, collate_fn
from model_vit import IntentNetViT, BasicBlock # Renamed model
from loss import DetectionIntentionLoss
from utils import generate_anchors

if __name__ == '__main__':
    # --- Configuration ---
    # USER_CONFIG: Update this path to your Argoverse 2 training data
    TRAIN_DATA_DIR = "./data/argoverse2/sensor/train" # Example placeholder
    # USER_CONFIG: Directory to save model checkpoints
    MODEL_SAVE_DIR_VIT = "./trained_models_vit" # Changed from final_models

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 8
    NUM_WORKERS = 0
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 10

    # Model & Loss Configuration
    USE_ROTATED_IOU = False
    APPLY_INTENTION_DOWNSAMPLING = True
    USE_INTENTION_WEIGHTS = False # If False, class_weights from data are not used by loss_fn

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    VIT_IMG_SIZE = (GRID_HEIGHT_PX, GRID_WIDTH_PX) # e.g., (224,224) or (720,400)

    VIT_BACKBONE_CFG = {
        'lidar_input_channels': LIDAR_TOTAL_CHANNELS,
        'map_input_channels': MAP_CHANNELS,
        'vit_model_name_lidar': 'vit_small_patch16_224', 
        'vit_model_name_map': 'vit_small_patch16_224',   
        'pretrained_lidar': False, # Set to True to use timm's pretrained weights (if available and compatible)
        'pretrained_map': False,
        'img_size': VIT_IMG_SIZE, # Crucial: must match input tensor size to ViT
        'drop_path_rate_lidar': 0.1,
        'drop_path_rate_map': 0.1,
        'lidar_adapter_out_channels': 192, # ViT Small embed_dim is 384, Tiny is 192. Adapter can change this.
        'map_adapter_out_channels': 192,   # This example uses 192, adjust based on your ViT and adapter.
        'fusion_block_planes': 512,        # Channels after fusion ResBlocks
        'fusion_block_layers': 2,          # Number of ResBlocks in fusion
        'fusion_block_kernel_size': 3,
        'fusion_block_stride': 1,          # Keep 1 if ViT patch already gives 16x, or 2 for 32x total
        'res_block_type': BasicBlock
    }
    try:
        # Assumes vit_model_name_lidar is like '..._patch<N>_...'
        vit_patch_stride_val = int(VIT_BACKBONE_CFG['vit_model_name_lidar'].split('_patch')[-1].split('_')[0])
    except ValueError:
        vit_patch_stride_val = 16 # Default if parsing fails
        print(f"Warning: Could not parse patch stride from ViT name, defaulting to {vit_patch_stride_val}.")
    FEATURE_MAP_STRIDE_VIT = vit_patch_stride_val * VIT_BACKBONE_CFG.get('fusion_block_stride', 1)

    print(f"--- ViT Training Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Training Data Directory: {TRAIN_DATA_DIR}")
    # print(f"AV2 Map API Available: {AV2_MAP_AVAILABLE}") # Printed by constants.py
    # print(f"Shapely Available: {SHAPELY_AVAILABLE}")   # Printed by constants.py
    print(f"BEV Image Size for ViT: {VIT_IMG_SIZE}")
    print(f"Using Rotated IoU: {USE_ROTATED_IOU}")
    print(f"Batch Size: {TRAIN_BATCH_SIZE}, Num Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Feature Map Stride (ViT): {FEATURE_MAP_STRIDE_VIT}")
    print(f"Apply Intention Downsampling: {APPLY_INTENTION_DOWNSAMPLING}")
    print(f"---------------------------------")

    # --- Validate Data Path ---
    train_data_path = Path(TRAIN_DATA_DIR)
    if not train_data_path.is_dir():
        print(f"ERROR: Training data directory not found: {TRAIN_DATA_DIR}")
        exit()

    # --- Create Dataset & DataLoader ---
    print("\nInitializing training dataset...")
    try:
        train_dataset = ArgoverseIntentNetDataset(data_dir=TRAIN_DATA_DIR, is_train=True)
        train_loader = DataLoader(
            train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=(DEVICE.type == 'cuda')
        )
        if len(train_loader) == 0:
            print("ERROR: Train DataLoader is empty. Check dataset or path.")
            exit()
        print(f"Training DataLoader initialized with {len(train_loader)} batches.")
    except Exception as e:
        print(f"ERROR initializing training dataset/loader: {e}")
        exit()

    # --- Calculate Intention Class Weights (Optional) ---
    intention_weights_tensor = None
    # (Identical weight calculation logic as in train_cnn.py - can be refactored into a utility if desired)
    if USE_INTENTION_WEIGHTS and APPLY_INTENTION_DOWNSAMPLING:
        print("Warning: Both USE_INTENTION_WEIGHTS and APPLY_INTENTION_DOWNSAMPLING are True. "
              "Downsampling will be applied; explicit weights will be ignored by the loss function.")
    elif USE_INTENTION_WEIGHTS:
        print("\nCalculating intention class weights from training data...")
        counts = np.zeros(NUM_INTENTION_CLASSES, dtype=np.int64)
        temp_loader_bs = max(16, TRAIN_BATCH_SIZE * 2)
        temp_loader = DataLoader(train_dataset, batch_size=temp_loader_bs, shuffle=False,
                                 num_workers=NUM_WORKERS, collate_fn=collate_fn)
        for batch_data in tqdm(temp_loader, desc="Calculating Weights", unit="batch"):
            if batch_data is None: continue
            for gt_item in batch_data['gt_list']:
                if 'intentions' in gt_item and gt_item['intentions'].numel() > 0:
                    intentions_np = gt_item['intentions'].long().cpu().numpy()
                    labels, c = np.unique(intentions_np[(intentions_np >= 0) & (intentions_np < NUM_INTENTION_CLASSES)], return_counts=True)
                    counts[labels] += c
        del temp_loader
        total_counts = counts.sum()
        if total_counts > 0:
            counts_smooth = counts + 1.0
            class_weights = total_counts / counts_smooth
            class_weights_normalized = class_weights / np.sum(class_weights)
            intention_weights_tensor = torch.tensor(class_weights_normalized, dtype=torch.float32).to(DEVICE)
            print("Calculated Intention Class Counts (Train Data):", counts)
            print("Calculated Normalized Intention Weights:", class_weights_normalized.round(4))
        else:
            print("Warning: No intention labels found in training data. Cannot calculate class weights.")
    print("---------------------------------")

    # --- Initialize Model, Loss, Optimizer ---
    print("\nInitializing ViT Model, Loss Function, and Optimizer...")
    model = IntentNetViT(backbone_cfg=VIT_BACKBONE_CFG).to(DEVICE) # Use renamed class

    loss_cfg_weights = intention_weights_tensor if USE_INTENTION_WEIGHTS and not APPLY_INTENTION_DOWNSAMPLING else None
    loss_fn = DetectionIntentionLoss(
        use_rotated_iou=USE_ROTATED_IOU,
        intention_class_weights=loss_cfg_weights,
        apply_intention_downsampling=APPLY_INTENTION_DOWNSAMPLING,
        dominant_intentions=DOMINANT_CLASSES_FOR_DOWNSAMPLING,
        intention_downsample_ratio=INTENTION_DOWNSAMPLE_RATIO
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False)

    # --- Generate Anchors ---
    print("\nGenerating anchors...")
    anchors = generate_anchors(
        bev_height=GRID_HEIGHT_PX, # Corrected parameter name
        bev_width=GRID_WIDTH_PX,   # Corrected parameter name
        feature_map_stride=FEATURE_MAP_STRIDE_VIT,
        anchor_configs=ANCHOR_CONFIGS_PAPER # Using the paper's W,L,R configs
    ).to(DEVICE)
    print(f"Anchors generated (stride {FEATURE_MAP_STRIDE_VIT}), shape: {anchors.shape}")
    print("---------------------------------")

    # --- Training Loop ---
    print("\n--- Starting ViT Training ---")
    # (Training loop is identical to train_cnn.py, just uses 'model' which is the ViT model here)
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_accum = 0.0
        epoch_cls_loss_accum, epoch_box_loss_accum, epoch_intent_loss_accum = 0.0, 0.0, 0.0
        batches_processed_epoch = 0

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} ViT Train", unit="batch")
        for batch_idx, batch_data in enumerate(progress_bar_train):
            if batch_data is None: continue

            lidar_bev = batch_data["lidar_bev"].to(DEVICE, non_blocking=True)
            map_bev = batch_data["map_bev"].to(DEVICE, non_blocking=True)
            gt_list = batch_data["gt_list"]

            optimizer.zero_grad()
            det_cls_logits, det_box_preds_rel, intent_logits = model(lidar_bev, map_bev)

            if torch.isnan(det_cls_logits).any() or torch.isnan(det_box_preds_rel).any() or torch.isnan(intent_logits).any():
                print(f"Warning: NaN detected in model output at batch {batch_idx+1}. Skipping batch.")
                continue

            loss_dict = loss_fn(det_cls_logits, det_box_preds_rel, intent_logits, anchors, gt_list)
            loss = loss_dict["loss"]

            if torch.isnan(loss):
                print(f"Warning: NaN loss calculated at batch {batch_idx+1}. Skipping batch. Loss dict: {loss_dict}")
                continue

            loss.backward()
            optimizer.step()

            epoch_loss_accum += loss.item()
            epoch_cls_loss_accum += loss_dict["cls_loss"].item()
            epoch_box_loss_accum += loss_dict["box_loss"].item()
            epoch_intent_loss_accum += loss_dict["intent_loss"].item()
            batches_processed_epoch += 1

            progress_bar_train.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Cls': f"{loss_dict['cls_loss'].item():.3f}",
                'Box': f"{loss_dict['box_loss'].item():.3f}",
                'Int': f"{loss_dict['intent_loss'].item():.3f}",
                '#Pos': loss_dict.get('num_pos_anchors', 'N/A').item() if isinstance(loss_dict.get('num_pos_anchors'), torch.Tensor) else loss_dict.get('num_pos_anchors', 'N/A')
            })

        if batches_processed_epoch > 0:
            avg_epoch_loss = epoch_loss_accum / batches_processed_epoch
            avg_cls_loss = epoch_cls_loss_accum / batches_processed_epoch
            avg_box_loss = epoch_box_loss_accum / batches_processed_epoch
            avg_intent_loss = epoch_intent_loss_accum / batches_processed_epoch
            print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_epoch_loss:.4f} "
                  f"(Cls: {avg_cls_loss:.4f}, Box: {avg_box_loss:.4f}, Intent: {avg_intent_loss:.4f}) "
                  f"LR: {optimizer.param_groups[0]['lr']:.1e}")
            scheduler.step(avg_epoch_loss)
        else:
            print(f"Epoch {epoch+1} Warning: No batches processed successfully.")

    print("\n--- ViT Training Finished ---")

    # Save the final ViT model
    save_dir = Path(MODEL_SAVE_DIR_VIT) 
    save_dir.mkdir(parents=True, exist_ok=True) 
    final_model_path = save_dir / "vit_model.pth" 
    torch.save({ 
    'epoch': NUM_EPOCHS, 
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(), 
    'backbone_cfg': VIT_BACKBONE_CFG, }, 
    final_model_path) 
    print(f"Saved final TRAINED ViT model to {final_model_path}")