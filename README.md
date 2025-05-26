# IntentNet Reimplementation and ViT Modification for Autonomous Driving Perception

---
This repository contains the code for a bachelor thesis project focused on rebuilding the IntentNet model and exploring a key architectural modification: replacing the original CNN-based feature extractors with Vision Transformer (ViT) based backbones for processing Bird's-Eye View (BEV) representations of LiDAR and map data. The primary goal is to perform joint vehicle detection and intention prediction in autonomous driving scenarios using the Argoverse 2 dataset.
## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Model Architectures](#model-architectures)
    *   [Original IntentNet (CNN-based)](#original-intentnet-cnn-based)
    *   [Modified IntentNet (ViT-based)](#modified-intentnet-vit-based)
3.  [Dataset](#dataset)
4.  [Features](#features)
5.  [File Structure](#file-structure)
6.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Dependencies](#dependencies)
    *   [Argoverse API and Shapely (Optional)](#argoverse-api-and-shapely-optional)
7.  [Data Preparation](#data-preparation)
    *   [Dataset Download](#dataset-download)
    *   [Pre-computing Intention Labels](#pre-computing-intention-labels)
8.  [Training](#training)
    *   [Training the CNN Model](#training-the-cnn-model)
    *   [Training the ViT Model](#training-the-vit-model)
9.  [Evaluation](#evaluation)
    *   [Evaluating the CNN Model](#evaluating-the-cnn-model)
    *   [Evaluating the ViT Model](#evaluating-the-vit-model)
10. [Key Configurations](#key-configurations)
11. [Results](#results)
    *   [Object Detection Performance](#object-detection-performance)
    *   [Intention Prediction Performance](#intention-prediction-performance)
12. [Future Work](#future-work)
13. [References](#references)

## Project Overview

IntentNet is a model designed for autonomous driving that jointly predicts the future intentions of other vehicles and detects them in the environment. This project first implements a version of IntentNet closely following the original paper's architecture, which utilizes Convolutional Neural Networks (CNNs) for feature extraction from BEV LiDAR and map inputs.

The core contribution of this thesis is the modification of this base IntentNet architecture. Specifically, the two CNN-based backbones (one for the LiDAR stream and one for the map stream) are replaced with two Vision Transformer (ViT) backbones. This exploration aims to evaluate the impact of using transformer-based architectures for feature extraction in this context.

## Model Architectures

Both models process BEV representations of LiDAR point clouds and HD map data. They consist of:
1.  A two-stream backbone (LiDAR and Map).
2.  A fusion module to combine features from both streams.
3.  Detection and Intention prediction heads.

### Original IntentNet (CNN-based)

*   Implemented in `model_cnn.py`.
*   This model uses a CNN backbone (ResNet-like blocks) for both the LiDAR and map data streams, as described in the IntentNet paper.
*   Features from both streams are fused and then fed into the detection and intention heads.

### Modified IntentNet (ViT-based)

*   Implemented in `model_vit.py`.
*   **Thesis Modification**: The CNN backbones for both LiDAR and map streams are replaced by Vision Transformer (ViT) models.
*   Patch embeddings from the ViTs are processed by adapter layers before fusion.
*   The fused features are then processed by similar detection and intention heads as the CNN version.
*   The stride of the fusion block after the ViT features is configurable (defaulted to 1 in this implementation, as ViT patch embeddings already achieve significant downsampling).
## Dataset

This project uses the **Argoverse 2 Sensor Dataset** using the split "Train Part 1" and "Validation Part 1".
*   **LiDAR Data**: Used to create BEV input. Multiple past LiDAR sweeps are aggregated.
*   **Map Data**: HD map information is rasterized into BEV channels.
*   **Annotations**: Ground truth 3D bounding boxes and vehicle categories. Intention labels are derived heuristically.

The `dataset.py` script handles data loading, BEV generation, and preparing samples for the models. Intention labels are pre-computed using `preprocess_intent_labels.py` which utilizes `heuristic_labeling.py`.
## Features

*   **Vehicle Detection**: Predicts 3D bounding boxes (center x, y, width, length, yaw) for vehicles in BEV.
*   **Intention Prediction**: Classifies the future intention of detected vehicles into 8 categories (e.g., Keep Lane, Turn Left, Turn Right, Left/Right Change Lane, Stopping/Stopped, Parked, Other).
*   **BEV Representation**: Converts LiDAR point clouds and HD map data into a multi-channel BEV grid.
*   **Modular Design**: Separate components for backbones, heads, loss functions, and data processing.
*   **Training and Evaluation Scripts**: Provided for both the CNN and ViT versions of the model.
*   **Data Augmentation**: Includes random flipping, rotation, scaling, and dropout for BEV inputs during training.

## File Structure
```
├── README.md # This file
├── constants.py # Global constants (BEV grid, anchors, intentions, etc.)
├── dataset.py # PyTorch Dataset class for Argoverse 2, data loading & BEV generation
├── heuristic_labeling.py # Heuristics for generating vehicle intention labels
├── preprocess_intent_labels.py # Script to pre-compute and cache intention labels
├── model_cnn.py # IntentNetCNN: CNN-based model implementation
├── model_vit.py # IntentNetViT: ViT-based model implementation (thesis modification)
├── heads.py # Shared detection and intention prediction head modules
├── loss.py # Combined loss function for detection and intention tasks
├── utils.py # Utility functions (anchor generation, NMS, IoU, augmentations)
├── train_cnn.py # Training script for the IntentNetCNN model
├── train_vit.py # Training script for the IntentNetViT model
├── eval_cnn.py # Evaluation script for the IntentNetCNN model
└──  eval_vit.py # Evaluation script for the IntentNetViT model
```

## Setup and Installation

### Prerequisites

*   Python 3.8+
*   PyTorch (tested with version >= 1.10, check your CUDA version compatibility)
*   pip for package installation
  
### Dependencies

It's recommended to set up a virtual environment:
```bash
python -m venv intentnet_env
source intentnet_env/bin/activate  # On Linux/macOS
# intentnet_env\Scripts\activate  # On Windows
```

Install the required packages. You can create a requirements.txt file with the following content:  
```
torch>=1.10.0
torchvision
numpy
pandas
pyarrow
scipy
opencv-python
tqdm
scikit-learn
timm           # For Vision Transformer models
# Optional dependencies (see below)
# shapely
# av2
```

Then install using pip:
```
pip install -r requirements.txt
```

### Argoverse API and Shapely (Optional)

*   **Argoverse API (`av2`)**: Required for `heuristic_labeling.py` to use map context for more accurate intention labels and for `dataset.py` if `AV2_MAP_AVAILABLE` is True (for map loading). Install from [Argoverse API GitHub](https://github.com/argoverse/av2-api).
*   **Shapely**: Used for calculating rotated IoU if `EVAL_USE_ROTATED_IOU` is True in evaluation scripts or `USE_ROTATED_IOU` is True in loss/training scripts. It can also enhance heuristic labeling.
    ```bash
    pip install shapely av2
    ```
The code includes checks (`AV2_MAP_AVAILABLE`, `SHAPELY_AVAILABLE` in `constants.py`) to allow running without these, potentially with reduced functionality (e.g., falling back to axis-aligned IoU or simpler heuristics).
## Data Preparation

### Dataset Download

1.  Download the Argoverse 2 Sensor Dataset from the [official website](https://www.argoverse.org/av2.html#sensor-link).
2.  Extract the dataset. You should have a directory structure like:
    ```
    /path/to/your/argoverse2/sensor/
    ├── train/
    │   ├── log_id_1/
    │   │   ├── annotations.feather
    │   │   ├── city_SE3_egovehicle.feather
    │   │   ├── map/
    │   │   │   └── log_map_archive_{log_id_1}.json
    │   │   └── sensors/
    │   │       └── lidar/
    │   │           ├── {timestamp_ns}.feather
    │   │           └── ...
    │   ├── log_id_2/
    │   └── ...
    ├── val/
    │   └── ...
    └── test/ (if using)
        └── ...
    ```
### Pre-computing Intention Labels

Before training, you need to pre-compute the heuristic intention labels for each scenario in your dataset splits. This will create an `annotations_with_intent.feather` file in each log directory.

Run the `preprocess_intent_labels.py` script:
```bash
python preprocess_intent_labels.py --data_root /path/to/your/argoverse2/sensor/ --splits train val --force
```
* **--data_root**: Path to the root of your Argoverse 2 sensor dataset (the directory containing train, val folders).
* **--splits**: List of splits to process (e.g., train val).
* **--force**: (Optional) Recompute labels even if annotations_with_intent.feather already exists.

This step can take a significant amount of time depending on your dataset size and CPU.

## Training
Update the TRAIN_DATA_DIR variable in the respective training scripts (train_cnn.py or train_vit.py) to point to your training data split (e.g., /path/to/your/argoverse2/sensor/train). Also, configure MODEL_SAVE_DIR_CNN or MODEL_SAVE_DIR_VIT for saving checkpoints.
### Training the CNN Model 
```
python train_cnn.py
```

Key configurations in train_cnn.py:
* **TRAIN_DATA_DIR**: Path to the training data.
* **MODEL_SAVE_DIR_CNN**: Directory to save model checkpoints.
* **TRAIN_BATCH_SIZE**, **NUM_EPOCHS**, **LEARNING_RATE**.
* **CNN_BACKBONE_CFG**: Configuration for the CNN backbone.
* **USE_ROTATED_IOU**, **APPLY_INTENTION_DOWNSAMPLING**: Loss function settings.
### Training the ViT Model
```
python train_vit.py
```

Key configurations in train_vit.py:
* **TRAIN_DATA_DIR**: Path to the training data.
* **MODEL_SAVE_DIR_VIT**: Directory to save model checkpoints.
* **TRAIN_BATCH_SIZE**, **NUM_EPOCHS**, **LEARNING_RATE**.
* **VIT_BACKBONE_CFG**: Configuration for the ViT backbone, including ViT model names, image size, adapter channels, and fusion block settings.
* **USE_ROTATED_IOU**, **APPLY_INTENTION_DOWNSAMPLING**: Loss function settings.
## Evaluation 
Update VAL_DATA_DIR and MODEL_SAVE_PATH_CNN or MODEL_SAVE_PATH_VIT in the evaluation scripts.
### Evaluating the CNN Model
```
python eval_cnn.py
```

Key configurations in eval_cnn.py:
* **VAL_DATA_DIR**: Path to the validation data (e.g., /path/to/your/argoverse2/sensor/val).
* **MODEL_SAVE_PATH_CNN**: Path to the trained CNN model checkpoint (.pth file).
* **EVAL_USE_ROTATED_IOU**: Whether to use rotated IoU for mAP calculation (requires Shapely).
## Evaluating the ViT Model
```
python eval_vit.py
```

Key configurations in eval_vit.py:
* **VAL_DATA_DIR**: Path to the validation data.
* **MODEL_SAVE_PATH_VIT**: Path to the trained ViT model checkpoint (.pth file).
* **EVAL_USE_ROTATED_IOU**: Whether to use rotated IoU for mAP calculation.

The evaluation scripts will output metrics such as mean Average Precision (mAP) for detection and accuracy/F1-scores for intention prediction.

## Key Configurations
Many important parameters are defined in constants.py. These include:

* BEV grid dimensions (GRID_HEIGHT_PX, GRID_WIDTH_PX, VOXEL_SIZE_M).
* LiDAR processing parameters (LIDAR_SWEEPS, LIDAR_HEIGHT_CHANNELS).
* Map channel definitions (MAP_CHANNELS).
* Anchor configurations (ANCHOR_CONFIGS_PAPER).
* Intention definitions and parameters (INTENTIONS_MAP, INTENTION_HORIZON_SECS).

Training and evaluation scripts also have their own specific configurations at the top of each file for paths, hyperparameters, and model choices.

## Results
The ViT-based model demonstrated improvement compared to the CNN-based model in detection and intention prediction. Based on experiments conducted on the "Validation Part 1" subset of Argoverse 2.
### Object Detection
The IntentNetViT model demonstrated significantly improved object detection capabilities.

* **IntentNetViT mAP@0.5**: 52%
  
* **IntentNetCNN mAP@0.5**: 34%

Detailed mAP scores (%) at Different IoU Thresholds:
| Model | IoU@0.5 | IoU@0.6 | IoU@0.7 | IoU@0.8 | IoU@0.9 |
|--------------------|---------|---------|---------|---------|---------|
| IntentNetCNN (Ours)| 34 | 22 | 18 | 4.1 | 0.7 |
| IntentNetViT (Ours)| 52 | 37 | 26 | 10.0| 1.3 |

### Intention Prediction Performance (for correctly detected vehicles, IoU > 0.5)
IntentNetViT also showed superior performance in intention prediction:

**IntentNetViT**:
* Macro F1-score: 0.51
* Overall Accuracy: 0.62

**IntentNetCNN**:
* Macro F1-score: 0.33
* Overall Accuracy: 0.46
It was also observed during training that the ViT model often exhibited a faster decrease in training loss, suggesting a potentially greater learning capacity or more efficient convergence on this dataset within the specified training regimen.

## Future Work
* Exploring advanced ViT backbones and fusion mechanisms.
* Integrating camera data.
* Explicit trajectory integration.
* Extended training regimens and full dataset utilization.
* Self-supervised pre-training for ViT backbones.
* Improved intention labeling and refined evaluation (rotated NMS/IoU).
* Qualitative analysis of attention mechanisms and detailed error analysis.
* Systematic efficiency benchmarking and model optimization for deployment.
* Explicit multi-agent modeling.

## References
Key academic papers referenced:
1. [3] Casas, S., Luo, W., & Urtasun, R. (2021). Intentnet: Learning to predict intention from raw sensor data. arXiv preprint arXiv:2101.07907.
2. [8] Wilson, B., Qi, W., Agarwal, T., et al. (2021). Argoverse 2: Next generation datasets for self-driving perception and forecasting. Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks.


