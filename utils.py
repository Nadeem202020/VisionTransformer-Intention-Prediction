import torch
import torchvision
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2
import random
import json

from constants import (GRID_HEIGHT_PX, GRID_WIDTH_PX, VOXEL_SIZE_M, BEV_PIXEL_OFFSET_X,
                       BEV_PIXEL_OFFSET_Y, Z_MIN, Z_MAX, LIDAR_HEIGHT_CHANNELS, LIDAR_SWEEPS,
                       MAP_CHANNELS, ANCHOR_CONFIGS_PAPER, INTENTIONS_MAP, VEHICLE_CATEGORIES, SHAPELY_AVAILABLE)

try:
    from shapely.geometry import Polygon
    _LOCAL_SHAPELY_POLYGON_AVAILABLE = True
except ImportError:
    _LOCAL_SHAPELY_POLYGON_AVAILABLE = False

def load_ego_poses(log_dir: str | Path) -> pd.DataFrame:
    """Loads ego vehicle poses from the specified log directory."""
    ego_file = Path(log_dir) / "city_SE3_egovehicle.feather"
    return feather.read_feather(ego_file)

def transform_points(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """Applies a 3D transformation matrix to an array of 3D points."""
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=points.dtype)
    homogeneous_points = np.hstack((points[:, :3], np.ones((points.shape[0], 1)))) 
    transformed_points = (transform_matrix @ homogeneous_points.T).T
    return transformed_points[:, :3]

def get_ego_centric_transform_matrix(ego_translation_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    """Computes the 2D transformation matrix from world to ego-centric BEV frame."""
    cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw) 
    rotation_matrix = np.array([[cos_yaw, -sin_yaw],
                                [sin_yaw,  cos_yaw]])
    translation_vector = -rotation_matrix @ ego_translation_xy 
    
    T = np.eye(3)
    T[:2, :2] = rotation_matrix
    T[:2, 2] = translation_vector
    return T

def world_to_bev_pixel(points_world_xy: np.ndarray, ego_tf_matrix: np.ndarray) -> np.ndarray:
    """Transforms world XY points to BEV pixel coordinates based on ego pose and BEV grid config."""
    if points_world_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=int)
    
    points_world_homo = np.hstack([points_world_xy, np.ones((points_world_xy.shape[0], 1))])
    points_ego_homo = (ego_tf_matrix @ points_world_homo.T).T 
    points_ego_xy = points_ego_homo[:, :2]

    pixel_x = BEV_PIXEL_OFFSET_X + points_ego_xy[:, 1] / VOXEL_SIZE_M
    pixel_y = BEV_PIXEL_OFFSET_Y - points_ego_xy[:, 0] / VOXEL_SIZE_M 
    
    pixel_coords = np.vstack([pixel_x, pixel_y]).T
    return np.round(pixel_coords).astype(int)

def create_intentnet_lidar_bev(points_list: list[np.ndarray | None],
                               intensity_list: list[np.ndarray | None],
                               num_expected_sweeps: int = LIDAR_SWEEPS) -> np.ndarray:
    """
    Creates a multi-sweep LiDAR BEV grid by stacking height-sliced intensity maps.
    Handles cases where fewer sweeps are provided by padding with zeros.
    """
    bev_shape = (LIDAR_HEIGHT_CHANNELS * num_expected_sweeps, GRID_HEIGHT_PX, GRID_WIDTH_PX)
    bev = np.zeros(bev_shape, dtype=np.float32)
    num_loaded_sweeps = min(len(points_list), len(intensity_list))

    for sweep_idx in range(num_loaded_sweeps):
        points, intensity = points_list[sweep_idx], intensity_list[sweep_idx]
        if points is None or intensity is None or points.shape[0] == 0:
            continue

        x_ego, y_ego, z_ego = points[:, 0], points[:, 1], points[:, 2]

        pixel_x = np.floor(BEV_PIXEL_OFFSET_X + y_ego / VOXEL_SIZE_M).astype(int)
        pixel_y = np.floor(BEV_PIXEL_OFFSET_Y - x_ego / VOXEL_SIZE_M).astype(int)

        valid_mask = (
            (pixel_x >= 0) & (pixel_x < GRID_WIDTH_PX) &
            (pixel_y >= 0) & (pixel_y < GRID_HEIGHT_PX) &
            (z_ego >= Z_MIN) & (z_ego < Z_MAX)
        )
        pixel_x, pixel_y, z_ego_filtered = pixel_x[valid_mask], pixel_y[valid_mask], z_ego[valid_mask]
        intensity_filtered = intensity[valid_mask]

        if len(pixel_x) == 0:
            continue

        z_slice_idx = np.floor((z_ego_filtered - Z_MIN) / (Z_MAX - Z_MIN) * LIDAR_HEIGHT_CHANNELS).astype(int)
        z_slice_idx = np.clip(z_slice_idx, 0, LIDAR_HEIGHT_CHANNELS - 1) 

        channel_offset = sweep_idx * LIDAR_HEIGHT_CHANNELS
        for h_idx in range(LIDAR_HEIGHT_CHANNELS):
            h_mask = (z_slice_idx == h_idx)
            if np.any(h_mask):
                np.maximum.at(
                    bev[channel_offset + h_idx],
                    (pixel_y[h_mask], pixel_x[h_mask]),
                    intensity_filtered[h_mask]
                )
    return bev

def rasterize_map_ego_centric(map_json_path: str, current_ego_pose: pd.Series) -> np.ndarray:
    """Rasterizes HD map elements into BEV channels relative to the current ego vehicle pose."""
    bev_map_shape = (MAP_CHANNELS, GRID_HEIGHT_PX, GRID_WIDTH_PX)
    try:
        with open(map_json_path, "r") as f:
            map_data = json.load(f)
    except Exception as e:
        print(f"Error loading map JSON {map_json_path}: {e}. Returning empty map.")
        return np.zeros(bev_map_shape, dtype=np.float32)

    lane_segments = map_data.get("lane_segments", {})
    crosswalks = map_data.get("pedestrian_crossings", {})

    ego_tx, ego_ty = current_ego_pose['tx_m'], current_ego_pose['ty_m']
    ego_q = current_ego_pose[['qx','qy','qz','qw']].values
    try:
        ego_yaw = R.from_quat(ego_q).as_euler('xyz')[2] 
    except ValueError:
        print(f"Warning: Invalid ego quaternion. Returning empty map for {map_json_path}")
        return np.zeros(bev_map_shape, dtype=np.float32)

    ego_tf_matrix_world_to_ego_bev = get_ego_centric_transform_matrix(np.array([ego_tx, ego_ty]), ego_yaw)

    def to_bev_pixel_local(points_world_xyz_list: list[dict]) -> np.ndarray:
        """Helper to transform list of world points to BEV pixel coordinates."""
        if not points_world_xyz_list: return np.empty((0, 2), dtype=int)
        valid_points = [p for p in points_world_xyz_list if isinstance(p, dict) and 'x' in p and 'y' in p]
        if not valid_points: return np.empty((0, 2), dtype=int)
        
        world_xy = np.array([[p['x'], p['y']] for p in valid_points])
        pixel_xy = world_to_bev_pixel(world_xy, ego_tf_matrix_world_to_ego_bev)
        
        valid_bev_mask = (
            (pixel_xy[:, 0] >= 0) & (pixel_xy[:, 0] < GRID_WIDTH_PX) &
            (pixel_xy[:, 1] >= 0) & (pixel_xy[:, 1] < GRID_HEIGHT_PX)
        )
        return pixel_xy[valid_bev_mask]

    bev_map_uint8 = np.zeros(bev_map_shape, dtype=np.uint8)

    for _, lane in lane_segments.items():
        left_px = to_bev_pixel_local(lane.get("left_lane_boundary", []))
        right_px = to_bev_pixel_local(lane.get("right_lane_boundary", []))

        if len(left_px) > 1 and len(right_px) > 1:
            polygon_points = np.vstack([left_px, np.flipud(right_px)])
            if polygon_points.shape[0] >= 3:
                cv2.fillPoly(bev_map_uint8[0], [polygon_points.reshape(-1, 1, 2)], color=1) 
                if lane.get("is_intersection", False):
                    cv2.fillPoly(bev_map_uint8[4], [polygon_points.reshape(-1, 1, 2)], color=1) 
                if lane.get("lane_type") == "BUS":
                    cv2.fillPoly(bev_map_uint8[5], [polygon_points.reshape(-1, 1, 2)], color=1) 

        if len(left_px) > 1:
            cv2.polylines(bev_map_uint8[1], [left_px.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=1)
        if len(right_px) > 1:
            cv2.polylines(bev_map_uint8[2], [right_px.reshape(-1, 1, 2)], isClosed=False, color=1, thickness=1)

        mark_channel_map = {"DASHED_WHITE": 6, "SOLID_WHITE": 7, "SOLID_YELLOW": 8}
        left_mark_type = lane.get("left_lane_mark_type", "")
        if left_mark_type in mark_channel_map and len(left_px) > 1:
            cv2.polylines(bev_map_uint8[mark_channel_map[left_mark_type]], [left_px.reshape(-1, 1, 2)], False, 1, 1)
        
        right_mark_type = lane.get("right_lane_mark_type", "")
        if right_mark_type in mark_channel_map and len(right_px) > 1:
            cv2.polylines(bev_map_uint8[mark_channel_map[right_mark_type]], [right_px.reshape(-1, 1, 2)], False, 1, 1)

    for _, cw in crosswalks.items():
        poly_world = cw.get('polygon', [])
        if poly_world:
            pts_px = to_bev_pixel_local(poly_world)
            if len(pts_px) >= 3:
                cv2.fillPoly(bev_map_uint8[3], [pts_px.reshape(-1, 1, 2)], color=1)

    return bev_map_uint8.astype(np.float32) 

def prepare_gt_for_frame(current_ts_ns: int, gt_df_with_intent: pd.DataFrame,
                         static_map) -> dict[str, torch.Tensor]:
    """
    Prepares ground truth bounding boxes (xywha) and intention labels for a given frame.
    Assumes gt_df_with_intent already contains pre-calculated 'heuristic_intent'.
    """
    frame_gt = gt_df_with_intent[
        (gt_df_with_intent['timestamp_ns'] == current_ts_ns) &
        (gt_df_with_intent['category'].isin(VEHICLE_CATEGORIES)) &
        (gt_df_with_intent['heuristic_intent'] != -1) 
    ]

    if 'heuristic_intent' not in frame_gt.columns:
        print(f"FATAL ERROR in prepare_gt_for_frame: 'heuristic_intent' column missing for timestamp {current_ts_ns}.")
        return {'boxes_xywha': torch.empty((0, 5), dtype=torch.float32),
                'intentions': torch.empty((0,), dtype=torch.long)}

    gt_boxes_xywha_list = []
    gt_intentions_list = []

    for _, box_row in frame_gt.iterrows():
        try:
            cx, cy = box_row['tx_m'], box_row['ty_m']
            w, l = abs(box_row['width_m']), abs(box_row['length_m'])
            quat = box_row[['qx', 'qy', 'qz', 'qw']].values
            heading_rad = R.from_quat(quat).as_euler('xyz', degrees=False)[2] 

            gt_boxes_xywha_list.append([cx, cy, w, l, heading_rad])
            gt_intentions_list.append(int(box_row['heuristic_intent']))
        except (ValueError, KeyError) as e:
            if len(gt_boxes_xywha_list) > len(gt_intentions_list):
                gt_boxes_xywha_list.pop()
            continue 

    if not gt_boxes_xywha_list: 
        return {'boxes_xywha': torch.empty((0, 5), dtype=torch.float32),
                'intentions': torch.empty((0,), dtype=torch.long)}

    return {
        'boxes_xywha': torch.tensor(gt_boxes_xywha_list, dtype=torch.float32),
        'intentions': torch.tensor(gt_intentions_list, dtype=torch.long)
    }

def decode_box_predictions(box_preds_rel: torch.Tensor, anchors_xywha: torch.Tensor) -> torch.Tensor:
    """
    Decodes relative box predictions (deltas) back to absolute BEV coordinates (cx, cy, w, l, heading).
    Args:
        box_preds_rel: Relative predictions [N, 6] -> (dx, dy, dw, dl, sin(d_h), cos(d_h)).
        anchors_xywha: Anchor boxes [N, 5] -> (cx_a, cy_a, w_a, l_a, heading_a).
    Returns:
        Absolute predicted boxes [N, 5] -> (cx, cy, w, l, heading).
    """
    if box_preds_rel.shape[0] == 0:
        return torch.empty((0, 5), device=box_preds_rel.device)

    eps = 1e-6 

    anchor_cx, anchor_cy, anchor_w, anchor_l, anchor_h_rad = \
        anchors_xywha[:, 0], anchors_xywha[:, 1], anchors_xywha[:, 2], anchors_xywha[:, 3], anchors_xywha[:, 4]
    
    dx, dy, dw, dl, d_sin, d_cos = \
        box_preds_rel[:, 0], box_preds_rel[:, 1], box_preds_rel[:, 2], \
        box_preds_rel[:, 3], box_preds_rel[:, 4], box_preds_rel[:, 5]

    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_l + anchor_cy 
    pred_w = torch.exp(dw) * anchor_w
    pred_l = torch.exp(dl) * anchor_l

    pred_heading_diff_rad = torch.atan2(d_sin, d_cos)
    pred_heading_rad = anchor_h_rad + pred_heading_diff_rad
    pred_heading_rad = torch.atan2(torch.sin(pred_heading_rad), torch.cos(pred_heading_rad))

    return torch.stack([pred_cx, pred_cy, pred_w, pred_l, pred_heading_rad], dim=-1)

def apply_nms(boxes_xywha: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.2) -> torch.Tensor:
    """
    Applies Non-Maximum Suppression (NMS) using torchvision.
    Converts xywha boxes to approximate x1y1x2y2 for axis-aligned NMS.
    """
    if boxes_xywha.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xywha.device)

    cx, cy, w, l = boxes_xywha[:, 0], boxes_xywha[:, 1], boxes_xywha[:, 2], boxes_xywha[:, 3]
    x1 = cx - w / 2
    y1 = cy - l / 2 
    x2 = cx + w / 2
    y2 = cy + l / 2
    boxes_corners_x1y1x2y2 = torch.stack([x1, y1, x2, y2], dim=1)

    return torchvision.ops.nms(boxes_corners_x1y1x2y2, scores, iou_threshold)

def compute_axis_aligned_iou(boxes1_xywh: torch.Tensor, boxes2_xywh: torch.Tensor) -> torch.Tensor:
    """Computes axis-aligned IoU between two sets of boxes (cx, cy, w, h)."""
    def to_corners(b: torch.Tensor) -> torch.Tensor: 
        x1, y1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
        x2, y2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    corners1, corners2 = to_corners(boxes1_xywh), to_corners(boxes2_xywh)
    inter_x1 = torch.maximum(corners1[:, None, 0], corners2[None, :, 0])
    inter_y1 = torch.maximum(corners1[:, None, 1], corners2[None, :, 1])
    inter_x2 = torch.minimum(corners1[:, None, 2], corners2[None, :, 2])
    inter_y2 = torch.minimum(corners1[:, None, 3], corners2[None, :, 3])
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = boxes1_xywh[:, 2] * boxes1_xywh[:, 3]
    area2 = boxes2_xywh[:, 2] * boxes2_xywh[:, 3]
    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / (union_area + 1e-7) 


def _xywha_to_shapely_polygon(box_xywha: np.ndarray) -> Polygon | None:
    """
    Converts a single (cx, cy, w, l, angle_rad) box to a Shapely Polygon.

    Angle Convention:
    - `cx`, `cy`: Center of the box.
    - `w`: Width of the box (dimension perpendicular to the angle_rad).
    - `l`: Length of the box (dimension aligned with the angle_rad).
    - `angle_rad`: Yaw angle in radians. This angle defines the orientation of the
                   LENGTH `l` of the box. A 0 radian angle means the length `l`
                   is aligned with the positive X-axis of the coordinate system
                   the box is defined in (e.g., ego-centric X-axis).
                   Positive angle is counter-clockwise.
    The local coordinate system of the unrotated box has its length `l` along the
    local y-axis and width `w` along the local x-axis. This system is then
    rotated by `angle_rad`.
    """
    if not _LOCAL_SHAPELY_POLYGON_AVAILABLE: 
        return None
        
    cx, cy, w, l, angle_rad = box_xywha
    
    hw, hl = w / 2.0, l / 2.0
    
    local_corners = np.array([
        [-hw, -hl],  
        [ hw, -hl],
        [ hw,  hl], 
        [-hw,  hl]  
    ])
    
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    R_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    rotated_corners = local_corners @ R_matrix.T
    global_corners = rotated_corners + np.array([cx, cy])
    
    try:
        poly = Polygon(global_corners)
        if not poly.is_valid:
            poly = poly.buffer(0) 
        return poly
    except Exception as e:
        return Polygon() 


def compute_rotated_iou(boxes1_xywha: torch.Tensor, boxes2_xywha: torch.Tensor) -> torch.Tensor:
    """
    Computes rotated IoU between two sets of boxes (cx, cy, w, l, angle_rad) using Shapely.
    `boxes1_xywha`: Tensor of shape [M, 5]
    `boxes2_xywha`: Tensor of shape [N, 5]
    Returns:
        iou_matrix: Tensor of shape [M, N]
    """
    if not SHAPELY_AVAILABLE:
        print("Warning: compute_rotated_iou called, but Shapely is not available (checked via constants.SHAPELY_AVAILABLE). Falling back to axis-aligned IoU.")
        return compute_axis_aligned_iou(boxes1_xywha[:, :4], boxes2_xywha[:, :4])

    np_boxes1 = boxes1_xywha.detach().cpu().numpy()
    np_boxes2 = boxes2_xywha.detach().cpu().numpy()

    num_boxes1 = np_boxes1.shape[0]
    num_boxes2 = np_boxes2.shape[0]
    iou_matrix = np.zeros((num_boxes1, num_boxes2), dtype=np.float32)

    if num_boxes1 == 0 or num_boxes2 == 0:
        return torch.from_numpy(iou_matrix).to(boxes1_xywha.device)

    polys1 = [_xywha_to_shapely_polygon(box) for box in np_boxes1]
    polys2 = [_xywha_to_shapely_polygon(box) for box in np_boxes2]
    
    areas1 = np.array([p.area if p and p.is_valid else 0 for p in polys1])
    areas2 = np.array([p.area if p and p.is_valid else 0 for p in polys2])

    for i in range(num_boxes1):
        if polys1[i] is None or not polys1[i].is_valid or areas1[i] < 1e-6:
            continue
        for j in range(num_boxes2):
            if polys2[j] is None or not polys2[j].is_valid or areas2[j] < 1e-6:
                continue

            try:
                poly_i = polys1[i] 
                poly_j = polys2[j] 


                intersection_area = poly_i.intersection(poly_j).area
                if intersection_area > 1e-7: 
                    union_area = areas1[i] + areas2[j] - intersection_area
                    if union_area > 1e-6:
                        iou_matrix[i, j] = intersection_area / union_area
            except Exception as e:
                iou_matrix[i, j] = 0.0

    return torch.from_numpy(iou_matrix).to(boxes1_xywha.device)

def random_flip_bev(lidar_bev: np.ndarray, map_bev: np.ndarray,
                    gt_boxes_xywha: np.ndarray, gt_intentions: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly flips BEV inputs horizontally (along Y-axis in ego frame)."""
    if random.random() < 0.5:
        lidar_bev = np.ascontiguousarray(np.flip(lidar_bev, axis=2)) 
        map_bev = np.ascontiguousarray(np.flip(map_bev, axis=2))
        if gt_boxes_xywha.shape[0] > 0:
            gt_boxes_xywha[:, 1] *= -1 
            gt_boxes_xywha[:, 4] *= -1  
            gt_boxes_xywha[:, 4] = np.arctan2(np.sin(gt_boxes_xywha[:, 4]), np.cos(gt_boxes_xywha[:, 4]))
        if gt_intentions.shape[0] > 0:
            mapping = {
                INTENTIONS_MAP["TURN_LEFT"]: INTENTIONS_MAP["TURN_RIGHT"],
                INTENTIONS_MAP["TURN_RIGHT"]: INTENTIONS_MAP["TURN_LEFT"],
                INTENTIONS_MAP["LEFT_CHANGE_LANE"]: INTENTIONS_MAP["RIGHT_CHANGE_LANE"],
                INTENTIONS_MAP["RIGHT_CHANGE_LANE"]: INTENTIONS_MAP["LEFT_CHANGE_LANE"]
            }
            original_intentions = gt_intentions.copy()
            for old_val, new_val in mapping.items():
                gt_intentions[original_intentions == old_val] = new_val
    return lidar_bev, map_bev, gt_boxes_xywha, gt_intentions

def random_rotate_bev(lidar_bev: np.ndarray, map_bev: np.ndarray, gt_boxes_xywha: np.ndarray,
                      angle_range_deg: tuple[float, float] = (-15.0, 15.0)
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly rotates BEV inputs and GT boxes around the ego vehicle's BEV center."""
    if random.random() < 0.5:
        angle_deg = random.uniform(angle_range_deg[0], angle_range_deg[1])
        angle_rad = np.radians(angle_deg)
        
        center_px_x = GRID_WIDTH_PX / 2.0
        center_px_y = GRID_HEIGHT_PX / 2.0 

        rot_mat_cv = cv2.getRotationMatrix2D((center_px_x, center_px_y), angle_deg, 1.0)

        def rotate_all_channels(bev_tensor: np.ndarray) -> np.ndarray:
            rotated_channels_list = [
                cv2.warpAffine(bev_tensor[i], rot_mat_cv, (GRID_WIDTH_PX, GRID_HEIGHT_PX),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                for i in range(bev_tensor.shape[0])
            ]
            return np.stack(rotated_channels_list, axis=0)

        lidar_bev = rotate_all_channels(lidar_bev)
        map_bev = rotate_all_channels(map_bev)

        if gt_boxes_xywha.shape[0] > 0:
            cx, cy = gt_boxes_xywha[:, 0].copy(), gt_boxes_xywha[:, 1].copy()
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad) 
            new_cx = cx * cos_a - cy * sin_a
            new_cy = cx * sin_a + cy * cos_a
            gt_boxes_xywha[:, 0], gt_boxes_xywha[:, 1] = new_cx, new_cy
            gt_boxes_xywha[:, 4] += angle_rad
            gt_boxes_xywha[:, 4] = np.arctan2(np.sin(gt_boxes_xywha[:, 4]), np.cos(gt_boxes_xywha[:, 4]))
    return lidar_bev, map_bev, gt_boxes_xywha

def random_scale_bev(lidar_bev: np.ndarray, map_bev: np.ndarray, gt_boxes_xywha: np.ndarray,
                     scale_range: tuple[float, float] = (0.95, 1.05)
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly scales BEV inputs and adjusts GT boxes accordingly."""
    if random.random() < 0.5:
        scale_factor = random.uniform(scale_range[0], scale_range[1])
        new_h, new_w = int(GRID_HEIGHT_PX * scale_factor), int(GRID_WIDTH_PX * scale_factor)

        def scale_all_channels(bev_tensor: np.ndarray) -> np.ndarray:
            scaled_channels_list = []
            for i in range(bev_tensor.shape[0]):
                resized_channel = cv2.resize(bev_tensor[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                final_channel = np.zeros((GRID_HEIGHT_PX, GRID_WIDTH_PX), dtype=bev_tensor.dtype)
                if scale_factor > 1.0: 
                    h_start = (new_h - GRID_HEIGHT_PX) // 2
                    w_start = (new_w - GRID_WIDTH_PX) // 2
                    final_channel = resized_channel[h_start:h_start+GRID_HEIGHT_PX, w_start:w_start+GRID_WIDTH_PX]
                else:
                    h_start = (GRID_HEIGHT_PX - new_h) // 2
                    w_start = (GRID_WIDTH_PX - new_w) // 2
                    final_channel[h_start:h_start+new_h, w_start:w_start+new_w] = resized_channel
                scaled_channels_list.append(final_channel)
            return np.stack(scaled_channels_list, axis=0)

        lidar_bev = scale_all_channels(lidar_bev)
        map_bev = scale_all_channels(map_bev)

        if gt_boxes_xywha.shape[0] > 0:
            gt_boxes_xywha[:, :4] *= scale_factor
    return lidar_bev, map_bev, gt_boxes_xywha

def random_bev_dropout(lidar_bev: np.ndarray, map_bev: np.ndarray,
                       dropout_prob: float = 0.1,
                       patch_size_range: tuple[int, int] = (20, 50),
                       num_patches_range: tuple[int, int] = (1, 5)
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Randomly zeros out rectangular patches in the BEV tensors."""
    if random.random() < dropout_prob:
        num_patches = random.randint(num_patches_range[0], num_patches_range[1])
        for _ in range(num_patches):
            patch_h = random.randint(patch_size_range[0], patch_size_range[1])
            patch_w = random.randint(patch_size_range[0], patch_size_range[1])
            start_y = random.randint(0, max(0, GRID_HEIGHT_PX - patch_h))
            start_x = random.randint(0, max(0, GRID_WIDTH_PX - patch_w))
            
            lidar_bev[:, start_y:start_y+patch_h, start_x:start_x+patch_w] = 0.0
            map_bev[:, start_y:start_y+patch_h, start_x:start_x+patch_w] = 0.0 
    return lidar_bev, map_bev

def augment_bev(lidar_bev: np.ndarray, map_bev: np.ndarray, gt_dict: dict
                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Applies a sequence of random BEV augmentations. Operates on NumPy arrays."""
    gt_boxes_np = gt_dict['boxes_xywha'].clone().numpy() if isinstance(gt_dict['boxes_xywha'], torch.Tensor) else gt_dict['boxes_xywha'].copy()
    gt_intentions_np = gt_dict['intentions'].clone().numpy() if isinstance(gt_dict['intentions'], torch.Tensor) else gt_dict['intentions'].copy()
    
    lidar_bev_aug, map_bev_aug = lidar_bev.copy(), map_bev.copy()

    lidar_bev_aug, map_bev_aug, gt_boxes_np, gt_intentions_np = random_flip_bev(lidar_bev_aug, map_bev_aug, gt_boxes_np, gt_intentions_np)
    lidar_bev_aug, map_bev_aug, gt_boxes_np = random_rotate_bev(lidar_bev_aug, map_bev_aug, gt_boxes_np)
    lidar_bev_aug, map_bev_aug, gt_boxes_np = random_scale_bev(lidar_bev_aug, map_bev_aug, gt_boxes_np)
    lidar_bev_aug, map_bev_aug = random_bev_dropout(lidar_bev_aug, map_bev_aug)

    augmented_gt_dict = {
        'boxes_xywha': torch.from_numpy(gt_boxes_np).float(),
        'intentions': torch.from_numpy(gt_intentions_np).long()
    }
    return lidar_bev_aug, map_bev_aug, augmented_gt_dict

def generate_anchors(bev_height: int = GRID_HEIGHT_PX, bev_width: int = GRID_WIDTH_PX,
                     feature_map_stride: int = 8,
                     anchor_configs: list[tuple] = ANCHOR_CONFIGS_PAPER,
                     voxel_size: float = VOXEL_SIZE_M,
                     offset_x_px: float = BEV_PIXEL_OFFSET_X, 
                     offset_y_px: float = BEV_PIXEL_OFFSET_Y  
                     ) -> torch.Tensor:
    """
    Generates anchor boxes across the BEV feature map.
    Anchors are defined in ego-centric metric coordinates (cx, cy, w, l, yaw_rad).
    """
    feature_map_h = bev_height // feature_map_stride
    feature_map_w = bev_width // feature_map_stride

    grid_y_fm, grid_x_fm = torch.meshgrid(torch.arange(feature_map_h), torch.arange(feature_map_w), indexing='ij')
    
    center_pixel_x = grid_x_fm * feature_map_stride + feature_map_stride / 2.0
    center_pixel_y = grid_y_fm * feature_map_stride + feature_map_stride / 2.0

    center_ego_y_m = (center_pixel_x - offset_x_px) * voxel_size
    center_ego_x_m = (offset_y_px - center_pixel_y) * voxel_size
    
    centers_m_ego = torch.stack([center_ego_x_m, center_ego_y_m], dim=-1)

    all_anchors_list = []
    for (w_anchor_m, l_anchor_m, yaw_anchor_rad) in anchor_configs:
        anchor_dims = torch.tensor([w_anchor_m, l_anchor_m, yaw_anchor_rad], dtype=torch.float32)
        anchor_dims_expanded = anchor_dims.view(1, 1, 3).expand(feature_map_h, feature_map_w, 3)
        
        anchors_at_config = torch.cat([centers_m_ego, anchor_dims_expanded], dim=-1)
        all_anchors_list.append(anchors_at_config.reshape(-1, 5))

    final_anchors = torch.cat(all_anchors_list, dim=0) 
    
    anchors_per_location = []
    for (w_m, l_m, r_rad) in anchor_configs:
        dims_tensor = torch.tensor([w_m, l_m, r_rad], dtype=torch.float32).unsqueeze(0).repeat(centers_m_ego.shape[0]*centers_m_ego.shape[1], 1) 
        current_anchors = torch.cat([centers_m_ego.reshape(-1,2), dims_tensor], dim=1) 
        anchors_per_location.append(current_anchors)
    
    stacked_anchors = torch.stack(anchors_per_location, dim=0)
    final_anchors_interleaved = stacked_anchors.transpose(0,1).reshape(-1,5)

    return final_anchors_interleaved

def calculate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Calculates Average Precision (AP) from recall and precision arrays (VOC PASCAL style)."""
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)