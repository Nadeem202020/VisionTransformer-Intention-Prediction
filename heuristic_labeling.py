import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd # Added for type hinting all_log_gt_boxes_df

from constants import (INTENTIONS_MAP, INTENTION_HORIZON_STEPS, MIN_SPEED_STOPPED,
                       MIN_SPEED_MOVING, HEADING_CHANGE_THRESH_TURN,
                       HEADING_CHANGE_THRESH_LANE_KEEP, PARKED_MAX_DISP_M,
                       KEEP_LANE_MAX_LAT_DIST_FALLBACK, AV2_MAP_AVAILABLE, SHAPELY_AVAILABLE)
# Note: map_search_radius is a default argument in the function, not from constants.py

def get_vehicle_intention_heuristic_enhanced(
    track_id: str,
    current_ts_ns: int,
    all_log_gt_boxes_df: pd.DataFrame,
    static_map, # Should be type hinted: ArgoverseStaticMap | None
    horizon_steps: int = INTENTION_HORIZON_STEPS,
    min_future_points: int = 5,
    stopped_speed_thresh: float = MIN_SPEED_STOPPED,
    moving_speed_thresh: float = MIN_SPEED_MOVING,
    turn_heading_thresh_rad: float = HEADING_CHANGE_THRESH_TURN,
    keep_heading_thresh_rad: float = HEADING_CHANGE_THRESH_LANE_KEEP,
    map_search_radius: float = 5.0, # Default value, not from constants
    parked_max_disp_m: float = PARKED_MAX_DISP_M,
    keep_lane_max_lat_dist_fallback: float = KEEP_LANE_MAX_LAT_DIST_FALLBACK
    ) -> int:
    """
    Derives a high-level vehicle intention label based on its future trajectory
    and (optionally) map context.
    """
    final_intent = INTENTIONS_MAP["OTHER"]

    vehicle_track = all_log_gt_boxes_df[all_log_gt_boxes_df['track_uuid'] == track_id].sort_values('timestamp_ns')
    current_idx_loc = vehicle_track.index[vehicle_track['timestamp_ns'] == current_ts_ns]

    if not current_idx_loc.any(): return final_intent
    current_row = vehicle_track.loc[current_idx_loc[0]]

    future_track = vehicle_track[vehicle_track['timestamp_ns'] > current_ts_ns].iloc[:horizon_steps]
    if len(future_track) < min_future_points: return final_intent

    start_pos_xy = np.array([current_row['tx_m'], current_row['ty_m']])
    end_pos_xy = future_track[['tx_m', 'ty_m']].values[-1]
    displacement_xy = end_pos_xy - start_pos_xy
    dist_traveled = np.linalg.norm(displacement_xy)
    time_elapsed_s = (future_track.iloc[-1]['timestamp_ns'] - current_ts_ns) * 1e-9 + 1e-9
    avg_speed = dist_traveled / time_elapsed_s

    try:
        start_heading_rad = R.from_quat(current_row[['qx', 'qy', 'qz', 'qw']].values).as_euler('xyz')[2]
        end_heading_rad = R.from_quat(future_track.iloc[-1][['qx', 'qy', 'qz', 'qw']].values).as_euler('xyz')[2]
        heading_change_rad = np.arctan2(np.sin(end_heading_rad - start_heading_rad),
                                      np.cos(end_heading_rad - start_heading_rad))
    except (ValueError, KeyError): return final_intent

    if avg_speed < stopped_speed_thresh:
        return INTENTIONS_MAP["PARKED"] if dist_traveled < parked_max_disp_m else INTENTIONS_MAP["STOPPING_STOPPED"]

    map_context_available = False
    is_intersection = False
    best_lane_id_for_context = None # Define for potential use in shapely check

    if AV2_MAP_AVAILABLE and static_map is not None:
        try:
            if abs(heading_change_rad) <= turn_heading_thresh_rad and avg_speed >= moving_speed_thresh:
                nearby_segments_info = static_map.get_nearby_lane_segments(start_pos_xy, map_search_radius)
                if nearby_segments_info:
                    min_dist_to_lane = float('inf')
                    for segment_id, dist_val in nearby_segments_info:
                        if dist_val < min_dist_to_lane:
                            min_dist_to_lane = dist_val
                            best_lane_id_for_context = segment_id
                    
                    if best_lane_id_for_context is not None:
                        current_lane_segment_obj = static_map.vector_lane_segments.get(best_lane_id_for_context)
                        if current_lane_segment_obj:
                            is_intersection = current_lane_segment_obj.is_intersection
                    map_context_available = True
        except Exception: map_context_available = False

    if avg_speed >= moving_speed_thresh:
        if heading_change_rad > turn_heading_thresh_rad: return INTENTIONS_MAP["TURN_LEFT"]
        if heading_change_rad < -turn_heading_thresh_rad: return INTENTIONS_MAP["TURN_RIGHT"]

    if map_context_available and is_intersection and avg_speed >= moving_speed_thresh:
        return INTENTIONS_MAP["KEEP_LANE"] if abs(heading_change_rad) <= keep_heading_thresh_rad else final_intent

    if avg_speed >= moving_speed_thresh and (not map_context_available or not is_intersection):
        if keep_heading_thresh_rad < abs(heading_change_rad) < turn_heading_thresh_rad:
            return INTENTIONS_MAP["LEFT_CHANGE_LANE"] if heading_change_rad > 0 else INTENTIONS_MAP["RIGHT_CHANGE_LANE"]

    if avg_speed >= moving_speed_thresh and abs(heading_change_rad) <= keep_heading_thresh_rad:
        can_check_polygons_shapely = False
        points_stay_in_valid_lanes_shapely = False
        if SHAPELY_AVAILABLE and map_context_available and not is_intersection and static_map is not None:
            current_lane_id_for_shapely = best_lane_id_for_context # Use previously found closest lane
            if current_lane_id_for_shapely is None: # If not found before, try again
                temp_nearby_info = static_map.get_nearby_lane_segments(start_pos_xy, map_search_radius)
                if temp_nearby_info:
                    min_d = float('inf');
                    for s_id, d_val in temp_nearby_info:
                        if d_val < min_d: min_d = d_val; current_lane_id_for_shapely = s_id
            
            if current_lane_id_for_shapely is not None:
                try:
                    from shapely.geometry import Point
                    from shapely.vectorized import contains as shapely_contains
                    successor_ids = static_map.get_lane_segment_successor_ids(current_lane_id_for_shapely) or set()
                    valid_lane_ids = {current_lane_id_for_shapely}.union(successor_ids)
                    valid_lane_polys = [p for p in [static_map.get_lane_segment_polygon(l_id) for l_id in valid_lane_ids] if p is not None and p.is_valid]
                    if valid_lane_polys:
                        can_check_polygons_shapely = True
                        future_points_geom = [Point(p_xy) for p_xy in future_track[['tx_m', 'ty_m']].values]
                        points_stay_in_valid_lanes_shapely = all(
                            any(shapely_contains(poly, pt)) for poly in valid_lane_polys for pt in future_points_geom)
                except Exception: pass # Defaults remain False

        if can_check_polygons_shapely and points_stay_in_valid_lanes_shapely:
            return INTENTIONS_MAP["KEEP_LANE"]
        elif not can_check_polygons_shapely: # Fallback if shapely check not possible or failed
            start_vec_xy = np.array([np.cos(start_heading_rad), np.sin(start_heading_rad)])
            lateral_dist_xy = np.linalg.norm(displacement_xy - np.dot(displacement_xy, start_vec_xy) * start_vec_xy)
            if lateral_dist_xy < keep_lane_max_lat_dist_fallback:
                return INTENTIONS_MAP["KEEP_LANE"]
                
    return final_intent