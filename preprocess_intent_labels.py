import argparse
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm # For overall progress and pandas integration
import traceback # For debugging, can be commented out in final version

# Project-specific imports (assuming all files are in the same root directory)
from constants import (AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE, VEHICLE_CATEGORIES,
                       INTENTIONS_MAP) # Import only what's needed by this script or its direct calls
from dataset import ScenarioValidator # Assuming dataset.py is in the same directory
from heuristic_labeling import get_vehicle_intention_heuristic_enhanced

# --- Configuration ---
OUTPUT_ANNOTATION_FILENAME = "annotations_with_intent.feather"

def preprocess_scenario(scenario_info: namedtuple, force_recompute: bool = False):
    """
    Processes a single scenario: loads annotations, calculates intentions, and saves.
    Returns True if successful, False otherwise.
    """
    log_dir = Path(scenario_info.log_dir)
    log_id = log_dir.name
    annotations_path = Path(scenario_info.annotations_path)
    map_json_path = Path(scenario_info.map_path)
    output_path = log_dir / OUTPUT_ANNOTATION_FILENAME

    if not force_recompute and output_path.exists():
        # print(f"  Skipping {log_id}: Output file already exists.")
        return "skipped"

    # print(f"Processing scenario {log_id}...")
    try:
        annotations_df = pd.read_feather(annotations_path)
        static_map = None

        if AV2_MAP_AVAILABLE:
            map_dir = map_json_path.parent
            if map_dir.is_dir() and any(map_dir.glob("log_map_archive_*.json")):
                # Conditional import if not already imported globally
                from av2.map.map_api import ArgoverseStaticMap
                static_map = ArgoverseStaticMap.from_map_dir(map_dir, build_raster=False)
            else:
                print(f"  Warning: Map data missing for {log_id}. Cannot reliably compute intentions requiring map context.")
                # Decide if you want to proceed with a None map or skip.
                # If map is critical for some heuristics, they should handle static_map being None.
        # else:
            # print(f"  Info: AV2 Map API not available. Proceeding without map context for {log_id}.")

        # Define a helper to apply to each row (track at a specific timestamp)
        def calculate_intent_for_row(row, full_log_df, current_static_map):
            if row['category'] in VEHICLE_CATEGORIES:
                return get_vehicle_intention_heuristic_enhanced(
                    track_id=row['track_uuid'],
                    current_ts_ns=row['timestamp_ns'],
                    all_log_gt_boxes_df=full_log_df, # Pass the full DataFrame for this log
                    static_map=current_static_map
                    # Other heuristic parameters will use defaults from heuristic_labeling.py or constants.py
                )
            return -1 # Default for non-vehicle categories or unhandled cases

        # Apply the heuristic. tqdm.pandas provides progress for the apply operation.
        # Ensure tqdm is registered with pandas if not done globally
        if not hasattr(pd.Series, 'progress_apply'): # Check if already registered
            tqdm.pandas(desc=f"Intent Calc {log_id[:8]}", leave=False)
        
        # Pass the entire annotations_df for the current log for history lookup
        annotations_df['heuristic_intent'] = annotations_df.progress_apply(
            lambda row: calculate_intent_for_row(row, annotations_df, static_map),
            axis=1
        )

        annotations_df.to_feather(output_path)
        # print(f"  Successfully processed and saved {log_id}.")
        return "processed"
    except Exception as e:
        print(f"  ERROR processing scenario {log_id}: {e}")
        # traceback.print_exc() # Uncomment for detailed debugging
        return "failed"

def main(data_root_dir: str, splits: list[str] = None, force_recompute: bool = False):
    """
    Main function to iterate over specified dataset splits and preprocess intention labels.
    """
    if splits is None:
        splits = ["train", "val"] # Default splits to process

    print(f"Starting intention label pre-computation.")
    print(f"Output file name per log: {OUTPUT_ANNOTATION_FILENAME}")
    print(f"Force recompute: {force_recompute}")

    overall_start_time = time.time()
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    for split_name in splits:
        print(f"\nProcessing split: {split_name}")
        # USER_CONFIG: Adjust this path structure if your data is organized differently
        # Assumes data_root_dir contains subfolders like 'train', 'val'
        # And each of those contains the scenario log folders.
        # Example: data_root_dir = "/path/to/argoverse2/sensor/"
        # Then split_dir will be "/path/to/argoverse2/sensor/train/"
        split_dir = Path(data_root_dir) / split_name
        if not split_dir.is_dir():
            print(f"  Directory for split '{split_name}' not found at: {split_dir}. Skipping.")
            continue

        validator = ScenarioValidator(str(split_dir), skip_known_corrupted=False)
        valid_scenarios = validator.find_valid_scenarios()

        if not valid_scenarios:
            print(f"  No valid scenarios found in {split_dir}.")
            continue
        
        print(f"  Found {len(valid_scenarios)} scenarios in {split_name} split.")
        
        split_processed = 0
        split_skipped = 0
        split_failed = 0

        for scenario_info in tqdm(valid_scenarios, desc=f"Processing {split_name}", unit="scenario"):
            result = preprocess_scenario(scenario_info, force_recompute)
            if result == "processed":
                split_processed += 1
            elif result == "skipped":
                split_skipped += 1
            else: # failed
                split_failed += 1
        
        print(f"  Finished {split_name} split: Processed={split_processed}, Skipped={split_skipped}, Failed={split_failed}")
        total_processed += split_processed
        total_skipped += split_skipped
        total_failed += split_failed

    overall_end_time = time.time()
    print(f"\nPre-computation finished for all specified splits.")
    print(f"  Total time taken: {(overall_end_time - overall_start_time) / 60:.2f} minutes")
    print(f"  Total Scenarios Processed Successfully: {total_processed}")
    print(f"  Total Scenarios Skipped (Already Done): {total_skipped}")
    print(f"  Total Scenarios Failed: {total_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute vehicle intention labels for Argoverse 2 dataset.")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Root directory of the Argoverse 2 sensor dataset (e.g., path to the directory containing 'train', 'val', 'test' splits)."
    )
    parser.add_argument(
        "--splits", nargs='+', default=["train", "val"],
        help="List of dataset splits to process (e.g., train val). Default is 'train' and 'val'."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-computation even if output files already exist."
    )
    args = parser.parse_args()

    main(data_root_dir=args.data_root, splits=args.splits, force_recompute=args.force)