import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from constants import (LIDAR_TOTAL_CHANNELS, MAP_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX,
                       NUM_INTENTION_CLASSES, ANCHOR_CONFIGS_PAPER,
                       AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE,
                       DOMINANT_CLASSES_FOR_DOWNSAMPLING, INTENTION_DOWNSAMPLE_RATIO)
from dataset import ArgoverseIntentNetDataset, collate_fn
from model_cnn import IntentNetCNN, BasicBlock 
from loss import DetectionIntentionLoss
from utils import generate_anchors

if __name__ == '__main__':
    TRAIN_DATA_DIR = "./data/argoverse2/sensor/train" 
    MODEL_SAVE_DIR_CNN = "./trained_models_cnn"

    TRAIN_BATCH_SIZE = 8
    NUM_WORKERS = 0
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 10 
    USE_ROTATED_IOU = False 
    APPLY_INTENTION_DOWNSAMPLING = True
    USE_INTENTION_WEIGHTS = False 

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CNN_BACKBONE_CFG = {
        'block': BasicBlock,
        'lidar_input_channels': LIDAR_TOTAL_CHANNELS,
        'map_input_channels': MAP_CHANNELS,
        'lidar_s1_planes': 160, 'lidar_s2_planes': 192, 'lidar_s3_planes': 224,
        'map_s1_planes': 32, 'map_s2_planes': 64, 'map_s3_planes': 96,
        'fusion_block_planes': 512, 'fusion_block_layers': 2,
        'num_blocks_per_stage': 2,
        'res_block2_kernel_size': 5, 'fusion_block_kernel_size': 3
    }
    FEATURE_MAP_STRIDE_CNN = 8 

    print(f"--- CNN Training Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Training Data Directory: {TRAIN_DATA_DIR}")
    print(f"Batch Size: {TRAIN_BATCH_SIZE}, Num Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print(f"Feature Map Stride (CNN): {FEATURE_MAP_STRIDE_CNN}")
    print(f"Apply Intention Downsampling: {APPLY_INTENTION_DOWNSAMPLING}")
    print(f"---------------------------------")

    train_data_path = Path(TRAIN_DATA_DIR)
    if not train_data_path.is_dir():
        print(f"ERROR: Training data directory not found: {TRAIN_DATA_DIR}")
        exit()

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

    intention_weights_tensor = None
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

    print("\nInitializing CNN Model, Loss Function, and Optimizer...")
    model = IntentNetCNN(backbone_cfg=CNN_BACKBONE_CFG).to(DEVICE) 

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

    print("\nGenerating anchors...")
    anchors = generate_anchors(
        bev_height=GRID_HEIGHT_PX,
        bev_width=GRID_WIDTH_PX,
        feature_map_stride=FEATURE_MAP_STRIDE_CNN,
        anchor_configs=ANCHOR_CONFIGS_PAPER
    ).to(DEVICE)
    print(f"Anchors generated (stride {FEATURE_MAP_STRIDE_CNN}), shape: {anchors.shape}")
    print("---------------------------------")

    print("\n--- Starting CNN Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss_accum = 0.0
        epoch_cls_loss_accum, epoch_box_loss_accum, epoch_intent_loss_accum = 0.0, 0.0, 0.0
        batches_processed_epoch = 0

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} CNN Train", unit="batch")
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

    print("\n--- CNN Training Finished ---")

    save_dir = Path(MODEL_SAVE_DIR_CNN) 
    save_dir.mkdir(parents=True, exist_ok=True) 
    final_model_path = save_dir / "cnn_model.pth" 
    torch.save({ 
    'epoch': NUM_EPOCHS, 
    'model_state_dict': model.state_dict(), 
    'optimizer_state_dict': optimizer.state_dict(), 
    'backbone_cfg': CNN_BACKBONE_CFG, }, 
    final_model_path) 
    print(f"Saved final TRAINED CNN model to {final_model_path}")