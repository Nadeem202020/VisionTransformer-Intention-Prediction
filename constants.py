import numpy as np


try:
    from av2.map.map_api import ArgoverseStaticMap
    AV2_MAP_AVAILABLE = True
except ImportError:
    AV2_MAP_AVAILABLE = False

try:
    from shapely.geometry import Polygon, Point 
    from shapely.vectorized import contains as shapely_contains
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


ANCHOR_CONFIGS_PAPER = [
    (2.0, 4.5, 0.0),        
    (2.0, 4.5, np.pi / 2),  
    (2.5, 2.5, 0.0),        
    (1.5, 9.0, 0.0),        
    (4.0, 2.0, 0.0),       
]
NUM_ANCHORS_PER_LOC = len(ANCHOR_CONFIGS_PAPER)


GRID_HEIGHT_PX, GRID_WIDTH_PX = 400, 720  
VOXEL_SIZE_M = 0.2                       
X_RANGE_M = GRID_HEIGHT_PX * VOXEL_SIZE_M 
Y_RANGE_M = GRID_WIDTH_PX * VOXEL_SIZE_M  


BEV_X_MIN, BEV_X_MAX = -X_RANGE_M / 4.0, X_RANGE_M * 3.0 / 4.0
BEV_Y_MIN, BEV_Y_MAX = -Y_RANGE_M / 2.0, Y_RANGE_M / 2.0


BEV_PIXEL_OFFSET_X = GRID_WIDTH_PX / 2.0  
BEV_PIXEL_OFFSET_Y = GRID_HEIGHT_PX * 3.0 / 4.0 


Z_MIN, Z_MAX = -2.0, 3.8                  
LIDAR_HEIGHT_CHANNELS = 29                
LIDAR_SWEEPS = 10                        
LIDAR_TOTAL_CHANNELS = LIDAR_HEIGHT_CHANNELS * LIDAR_SWEEPS 

MAP_CHANNELS = 9


NUM_INTENTION_CLASSES = 8                 
INTENTION_HORIZON_SECS = 3.0              
INTENTION_HORIZON_STEPS = int(INTENTION_HORIZON_SECS * 10) 

MIN_SPEED_STOPPED = 0.5                   
MIN_SPEED_MOVING = 1.0                   

HEADING_CHANGE_THRESH_TURN = np.radians(20) 
HEADING_CHANGE_THRESH_LANE_KEEP = np.radians(5) 

PARKED_MAX_DISP_M = 0.5                   
KEEP_LANE_MAX_LAT_DIST_FALLBACK = 0.5     


INTENTIONS_MAP = {
    "KEEP_LANE": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2,
    "LEFT_CHANGE_LANE": 3, "RIGHT_CHANGE_LANE": 4,
    "STOPPING_STOPPED": 5, "PARKED": 6, "OTHER": 7
}
INTENTIONS_MAP_REV = {v: k for k, v in INTENTIONS_MAP.items()}


DOMINANT_CLASSES_FOR_DOWNSAMPLING = {
    INTENTIONS_MAP["KEEP_LANE"],
    INTENTIONS_MAP["OTHER"],        
    INTENTIONS_MAP["PARKED"],
}
INTENTION_DOWNSAMPLE_RATIO = 0.85 


VEHICLE_CATEGORIES = {
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK",
    "MOTORCYCLE", "SCHOOL_BUS", "ARTICULATED_BUS", "VEHICULAR_TRAILER",
    "TRUCK_CAB", "BICYCLE", "BICYCLIST", "MOTORCYCLIST"
}
