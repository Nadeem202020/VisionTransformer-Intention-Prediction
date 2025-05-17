import torch
from torch.utils.data import Dataset # DataLoader is not used directly in this file
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation as R
import traceback
import time
import os
from collections import namedtuple
# cv2 is imported in utils.py where rasterize_map_ego_centric is defined

# Assuming all custom modules are in the same root directory
from constants import (LIDAR_SWEEPS, AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE,
                       GRID_HEIGHT_PX, GRID_WIDTH_PX, VOXEL_SIZE_M, BEV_X_MIN, BEV_X_MAX,
                       BEV_Y_MIN, BEV_Y_MAX, BEV_PIXEL_OFFSET_X, BEV_PIXEL_OFFSET_Y, Z_MIN, Z_MAX,
                       LIDAR_HEIGHT_CHANNELS, LIDAR_TOTAL_CHANNELS, MAP_CHANNELS,
                       VEHICLE_CATEGORIES) # Ensure all needed constants are imported
from utils import (load_ego_poses, transform_points, # Removed unused: get_ego_centric_transform_matrix, world_to_bev_pixel
                   create_intentnet_lidar_bev, rasterize_map_ego_centric,
                   prepare_gt_for_frame, augment_bev)
from heuristic_labeling import get_vehicle_intention_heuristic_enhanced


class ScenarioValidator:
    """
    Validates scenario directories to ensure all required files are present
    and optionally skips known corrupted logs.
    """
    def __init__(self, base_path: str, skip_known_corrupted: bool = True, min_feather_size_bytes: int = 1024):
        self.base_path = Path(base_path)
        self.ScenarioPaths = namedtuple("ScenarioPaths", ["log_dir", "map_path", "annotations_path"])
        self.skip_known_corrupted = skip_known_corrupted
        self.min_feather_size_bytes = min_feather_size_bytes
        # List of known corrupted log directory names
        self.KNOWN_CORRUPTED_LOGS = {
            # Add log IDs if any are identified as consistently problematic
        }
        # Optional: Could print these for debugging during init
        # print(f"ScenarioValidator: Skip known corrupted logs: {self.skip_known_corrupted}")
        # print(f"ScenarioValidator: Known corrupted logs to skip: {self.KNOWN_CORRUPTED_LOGS}")

    def find_valid_scenarios(self) -> list[namedtuple]:
        """
        Scans the base_path for valid scenario directories.
        Returns:
            list: A list of ScenarioPaths namedtuples for valid scenarios.
        """
        valid_scenarios = []
        print(f"ScenarioValidator: Searching for scenarios in: {self.base_path}")
        if not self.base_path.is_dir(): # Corrected from self.base_path.exists() to is_dir() for clarity
            print(f"Error: Base path does not exist or is not a directory: {self.base_path}")
            return []

        total_dirs_scanned = 0
        skipped_corrupted = 0
        skipped_reasons = {}

        start_time = time.time()
        try:
            # os.scandir can be more efficient for listing many directories
            iterator = os.scandir(self.base_path)
        except OSError as e:
            print(f"Error: Cannot scan directory {self.base_path}: {e}")
            return []

        for entry in iterator:
            if not entry.is_dir():
                continue

            scenario_dir = Path(entry.path)
            total_dirs_scanned += 1
            scenario_name = scenario_dir.name

            if total_dirs_scanned > 0 and total_dirs_scanned % 50 == 0:
                print(f"  Scanned {total_dirs_scanned} directories...")

            if self.skip_known_corrupted and scenario_name in self.KNOWN_CORRUPTED_LOGS:
                skipped_corrupted += 1
                continue

            validation_result = self._validate_scenario(scenario_dir)
            if isinstance(validation_result, self.ScenarioPaths):
                valid_scenarios.append(validation_result)
            elif isinstance(validation_result, str):
                reason = validation_result
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        end_time = time.time()
        print(f"\nScenario Scan Summary (took {end_time - start_time:.2f} seconds):")
        print(f"  Total directories scanned: {total_dirs_scanned}")
        if self.skip_known_corrupted:
            print(f"  Skipped (known corrupted): {skipped_corrupted}")
        if skipped_reasons:
            print(f"  Skipped (invalid structure/files): {sum(skipped_reasons.values())}")
            print("  Skip Reasons:")
            for reason, count in skipped_reasons.items():
                print(f"    - {reason}: {count}")
        print(f"  Found {len(valid_scenarios)} valid scenarios for processing.")
        return valid_scenarios

    def _validate_scenario(self, scenario_dir: Path):
        """Checks a single scenario directory for required files and basic integrity."""
        lidar_dir = scenario_dir / "sensors" / "lidar"
        annotation_file = scenario_dir / "annotations.feather"
        map_dir = scenario_dir / "map"
        ego_pose_file = scenario_dir / "city_SE3_egovehicle.feather"
        log_id = scenario_dir.name

        required_paths = {
            "Lidar directory": lidar_dir,
            "Annotations file": annotation_file,
            "Map directory": map_dir,
            "Ego pose file": ego_pose_file,
        }
        for name, path_obj in required_paths.items():
            if (path_obj.is_dir() and not any(path_obj.iterdir())) or \
               (path_obj.is_file() and path_obj.stat().st_size < self.min_feather_size_bytes and self.min_feather_size_bytes > 0) or \
               not path_obj.exists():
                return f"Missing or invalid {name.lower()} ({path_obj.name})"

        if not any(lidar_dir.glob("*.feather")):
            return "No *.feather files in lidar directory"

        map_files = list(map_dir.glob(f"log_map_archive_{log_id}*.json"))
        if not map_files:
            map_files_fallback = list(map_dir.glob("log_map_archive_*.json"))
            if not map_files_fallback:
                return f"No 'log_map_archive_{log_id}*.json' or generic map file found in map directory"
            map_files = map_files_fallback
            # print(f"  Warning: Used fallback map glob for {log_id}") # Keep for debugging if needed

        return self.ScenarioPaths(
            log_dir=str(scenario_dir),
            map_path=str(map_files[0]),
            annotations_path=str(annotation_file)
        )


def collate_fn(batch: list) -> dict | None:
    """
    Custom collate function to handle batches of data items,
    especially filtering out None items that may result from errors in __getitem__.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    lidar_bevs = torch.stack([item["lidar_bev"] for item in batch])
    map_bevs = torch.stack([item["map_bev"] for item in batch])
    gt_list = [item["gt"] for item in batch] # Ground truth kept as a list of dictionaries

    return {"lidar_bev": lidar_bevs, "map_bev": map_bevs, "gt_list": gt_list}


class ArgoverseIntentNetDataset(Dataset):
    """
    Dataset class for Argoverse 2, preparing data for IntentNet-like models.
    Handles loading of LiDAR sweeps, map data, and ground truth annotations.
    """
    def __init__(self, data_dir: str, num_sweeps: int = LIDAR_SWEEPS, is_train: bool = False):
        # data_dir: Path to the root directory of the dataset split (e.g., train or val).
        self.data_dir = Path(data_dir)
        self.num_sweeps = num_sweeps
        self.is_train = is_train

        validator = ScenarioValidator(str(self.data_dir))
        self.valid_scenario_paths = validator.find_valid_scenarios()
        if not self.valid_scenario_paths:
            raise ValueError(f"No valid scenarios found in {self.data_dir}. Please check the path and data integrity.")

        self.log_data_cache = {}  # Cache for frequently accessed log data (ego poses, GT annotations)
        self.sequences = self._create_sequences()
        if not self.sequences:
            raise ValueError(f"Could not create any valid sequences from the provided scenarios in {self.data_dir}.")
        print(f"Dataset Initialized: {'Train' if is_train else 'Validation'}. Found {len(self.sequences)} sequences.")

    def _create_sequences(self) -> list[dict]:
        """Generates a list of all valid (current_timestamp, past_sweeps) sequences from scenarios."""
        sequences = []
        print("Creating sequences from valid scenarios...")
        for scenario_info in self.valid_scenario_paths:
            log_dir = Path(scenario_info.log_dir)
            log_id = log_dir.name
            lidar_dir = log_dir / "sensors" / "lidar"

            try:
                if not lidar_dir.is_dir(): # Should be caught by validator, but as a safeguard
                    print(f"  Warning (_create_sequences): LiDAR directory missing for {log_id}. Skipping.")
                    continue

                # Timestamps are filenames without extension
                timestamps = sorted([int(p.stem) for p in lidar_dir.glob("*.feather")])

                if len(timestamps) < self.num_sweeps:
                    continue # Not enough sweeps to form a sequence

                # Create sequences by sliding a window of size num_sweeps
                for i in range(len(timestamps) - self.num_sweeps + 1):
                    current_ts_ns = timestamps[i + self.num_sweeps - 1] # The last sweep in the window is current
                    sweep_ts_list = timestamps[i : i + self.num_sweeps]   # List of all sweeps in the window
                    sequences.append({
                        "log_id": log_id,
                        "log_dir": str(log_dir), # Store as string for easier handling
                        "map_json_path": scenario_info.map_path,
                        "annotations_path": scenario_info.annotations_path, # Path to original annotations
                        "current_ts_ns": current_ts_ns,
                        "sweep_ts_list": sweep_ts_list
                    })
            except ValueError as e: # If p.stem cannot be converted to int
                print(f"  Warning (_create_sequences): Error converting timestamp filename in {log_id}: {e}. Skipping.")
            except Exception as e: # Catch-all for other unexpected errors during sequence creation
                print(f"  Warning (_create_sequences): Unexpected error processing log {log_id}: {e}. Skipping.")
        print(f"Created {len(sequences)} sequences in total.")
        return sequences

    def _get_log_data(self, log_id: str, log_dir: str, annotations_path: str) -> dict | None:
        """
        Loads and caches essential data for a given log_id.
        This includes ego poses and ground truth annotations (with pre-computed intentions).
        """
        log_dir_path = Path(log_dir)
        # Path to the pre-computed annotation file with intention labels
        intent_annotation_filename = "annotations_with_intent.feather"
        intent_annotation_file_path = log_dir_path / intent_annotation_filename

        if log_id not in self.log_data_cache:
            # print(f"    CACHE MISS for log: {log_id}. Loading...") # Optional debug print
            try:
                if not intent_annotation_file_path.is_file():
                    # This indicates a preprocessing step was missed or failed.
                    # The heuristic labeling should produce this file for each log.
                    error_msg = (f"FATAL ERROR: Pre-computed intent file missing for log {log_id} "
                                 f"at {intent_annotation_file_path}. "
                                 "Please run the intention pre-computation script.")
                    print(error_msg)
                    # Raising an error might be too disruptive if only some logs miss it.
                    # Consider returning None and letting __getitem__ handle it,
                    # or making the pre-computation script a hard prerequisite.
                    # For now, we'll let it try to proceed and fail in __getitem__ if data is None.
                    self.log_data_cache[log_id] = None # Mark as failed to load
                    return None


                gt_df_with_intent = pd.read_feather(intent_annotation_file_path)
                ego_poses_df = load_ego_poses(log_dir_path) # From utils.py

                map_api = None
                if AV2_MAP_AVAILABLE: # Constant from constants.py
                    map_base_path = log_dir_path / "map"
                    if map_base_path.is_dir() and any(map_base_path.iterdir()):
                        from av2.map.map_api import ArgoverseStaticMap # Local import
                        map_api = ArgoverseStaticMap.from_map_dir(map_base_path, build_raster=False)

                self.log_data_cache[log_id] = {
                    "ego_poses": ego_poses_df,
                    "gt_df": gt_df_with_intent, # This now includes 'heuristic_intent'
                    "map_api": map_api
                }
                # print(f"      -> Loaded and cached {log_id}") # Optional debug print

            except FileNotFoundError as fnf_err:
                print(f"Error in _get_log_data (FileNotFound) for {log_id}: {fnf_err}")
                self.log_data_cache[log_id] = None
            except Exception as e:
                print(f"Error loading data cache for log {log_id}: {e}")
                traceback.print_exc()
                self.log_data_cache[log_id] = None
        # else:
            # print(f"    CACHE HIT for log: {log_id}") # Optional debug print

        return self.log_data_cache.get(log_id) # Returns None if log_id was not successfully cached

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict | None:
        """
        Retrieves a single data sample (LiDAR BEV, Map BEV, GT) for the given index.
        """
        if not (0 <= idx < len(self.sequences)):
            raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.sequences)}")

        sequence_info = self.sequences[idx]
        log_id = sequence_info["log_id"]
        log_dir = sequence_info["log_dir"] # Already a string
        map_json_path = sequence_info["map_json_path"] # Already a string
        # annotations_path = sequence_info["annotations_path"] # Path to original, not pre-computed
        current_ts_ns = sequence_info["current_ts_ns"]
        sweep_ts_list = sequence_info["sweep_ts_list"]

        try:
            log_data = self._get_log_data(log_id, log_dir, sequence_info["annotations_path"])
            if log_data is None:
                print(f"Warning: No log data found for log_id {log_id} in __getitem__ for index {idx}. Skipping item.")
                return None
            ego_poses_df = log_data["ego_poses"]
            gt_df_with_intent = log_data["gt_df"]
            map_api = log_data["map_api"]

            current_ego_pose_row = ego_poses_df[ego_poses_df['timestamp_ns'] == current_ts_ns]
            if current_ego_pose_row.empty:
                # print(f"Warning: Current ego pose not found for ts {current_ts_ns} in log {log_id}. Skipping item.")
                return None
            current_ego_pose = current_ego_pose_row.iloc[0]

            tx, ty, tz = current_ego_pose['tx_m'], current_ego_pose['ty_m'], current_ego_pose['tz_m']
            qx, qy, qz, qw = current_ego_pose['qx'], current_ego_pose['qy'], current_ego_pose['qz'], current_ego_pose['qw']
            try:
                rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            except ValueError: # Handle invalid quaternion
                print(f"Warning: Invalid quaternion for ego pose at ts {current_ts_ns} in log {log_id}. Skipping item.")
                return None

            world_SE3_ego = np.eye(4)
            world_SE3_ego[:3, :3] = rot_mat
            world_SE3_ego[:3, 3] = [tx, ty, tz]
            ego_SE3_world = np.linalg.inv(world_SE3_ego)

            points_list, intensity_list = [], []
            lidar_base_path = Path(log_dir) / "sensors" / "lidar"
            for ts_sweep in sweep_ts_list:
                sweep_path = lidar_base_path / f"{ts_sweep}.feather"
                if not sweep_path.is_file():
                    points_list.append(None); intensity_list.append(None)
                    continue
                try:
                    sweep_df = pd.read_feather(sweep_path, columns=['x', 'y', 'z', 'intensity'])
                    if sweep_df.empty:
                        points_list.append(None); intensity_list.append(None)
                        continue
                except pyarrow.ArrowInvalid: # Catch specific error for corrupt feather files
                    print(f"Warning: Corrupt LiDAR sweep file: {sweep_path}. Skipping sweep.")
                    points_list.append(None); intensity_list.append(None)
                    continue

                pts_world = sweep_df[['x', 'y', 'z']].values
                intensity = sweep_df['intensity'].values.astype(np.float32) # Ensure intensity is float

                sweep_pose_row = ego_poses_df[ego_poses_df['timestamp_ns'] == ts_sweep]
                if sweep_pose_row.empty:
                    points_list.append(None); intensity_list.append(None)
                    continue

                sw_tx, sw_ty, sw_tz = sweep_pose_row.iloc[0]['tx_m'], sweep_pose_row.iloc[0]['ty_m'], sweep_pose_row.iloc[0]['tz_m']
                sw_q = sweep_pose_row.iloc[0][['qx','qy','qz','qw']].values
                try:
                    sw_rot = R.from_quat(sw_q).as_matrix()
                except ValueError:
                    points_list.append(None); intensity_list.append(None)
                    continue # Skip sweep if its pose is invalid

                sw_tf_world_ego = np.eye(4)
                sw_tf_world_ego[:3,:3] = sw_rot
                sw_tf_world_ego[:3,3] = [sw_tx, sw_ty, sw_tz]
                rel_tf = ego_SE3_world @ sw_tf_world_ego
                pts_curr_ego = transform_points(pts_world, rel_tf)
                points_list.append(pts_curr_ego)
                intensity_list.append(intensity)

            if all(p is None for p in points_list): # If all sweeps failed to load
                # print(f"Warning: All LiDAR sweeps failed to load for ts {current_ts_ns} in log {log_id}. Skipping item.")
                return None

            lidar_bev_np = create_intentnet_lidar_bev(points_list, intensity_list)
            map_bev_np = rasterize_map_ego_centric(map_json_path, current_ego_pose)

            frame_gt_dict = prepare_gt_for_frame(current_ts_ns, gt_df_with_intent, map_api)
            # prepare_gt_for_frame now returns a default empty dict if no GT, so direct check is fine
            # if frame_gt_dict is None:
            #     frame_gt_dict = {'boxes_xywha': torch.empty((0, 5), dtype=torch.float32),
            #                      'intentions': torch.empty((0,), dtype=torch.long)}

            if self.is_train:
                lidar_bev_np, map_bev_np, frame_gt_dict = augment_bev(lidar_bev_np, map_bev_np, frame_gt_dict)

            final_lidar_bev = torch.from_numpy(lidar_bev_np).float()
            final_map_bev = torch.from_numpy(map_bev_np).float()
            # Ensure GT tensors are on CPU before collate_fn (though they are created from numpy, so should be)
            final_gt_dict = {
                'boxes_xywha': frame_gt_dict['boxes_xywha'].float(),
                'intentions': frame_gt_dict['intentions'].long()
            }
            return {"lidar_bev": final_lidar_bev, "map_bev": final_map_bev, "gt": final_gt_dict}

        except Exception as e:
            # Print more context for debugging
            print(f"!!! UNHANDLED ERROR in __getitem__ for index {idx}, log_id {log_id}, timestamp {current_ts_ns} !!!")
            print(f"Sequence Info: {self.sequences[idx] if idx < len(self.sequences) else 'Index out of bounds in error handler'}")
            traceback.print_exc()
            return None