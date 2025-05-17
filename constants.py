import numpy as np

# --- Optional Dependency Checks ---
# These flags indicate whether the respective libraries were successfully imported
# and can be used by other modules (e.g., for map processing or advanced heuristics).
try:
    from av2.map.map_api import ArgoverseStaticMap
    AV2_MAP_AVAILABLE = True
except ImportError:
    AV2_MAP_AVAILABLE = False

try:
    from shapely.geometry import Polygon, Point # Point might not be directly used by constants but good for the check
    from shapely.vectorized import contains as shapely_contains
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

# --- Anchor Configuration ---
# Dimensions: (Width, Length, Orientation in Radians)
ANCHOR_CONFIGS_PAPER = [
    (2.0, 4.5, 0.0),        # Base car-like ratio, forward
    (2.0, 4.5, np.pi / 2),  # Base car-like ratio, sideways
    (2.5, 2.5, 0.0),        # Square-ish (e.g., small vans, some SUVs)
    (1.5, 9.0, 0.0),        # Long and narrow (e.g., trailers, some buses)
    (4.0, 2.0, 0.0),        # Wide and short (less common, but for completeness)
]
NUM_ANCHORS_PER_LOC = len(ANCHOR_CONFIGS_PAPER)

# --- BEV Grid Configuration ---
GRID_HEIGHT_PX, GRID_WIDTH_PX = 400, 720  # BEV image dimensions in pixels
VOXEL_SIZE_M = 0.2                        # Size of each pixel in meters
X_RANGE_M = GRID_HEIGHT_PX * VOXEL_SIZE_M # Total height of BEV grid in meters
Y_RANGE_M = GRID_WIDTH_PX * VOXEL_SIZE_M  # Total width of BEV grid in meters

# BEV grid boundaries in ego-vehicle coordinate system (meters)
# X-axis is forward, Y-axis is left. Origin is at the ego vehicle's rear axle.
# Grid is centered with a forward bias.
BEV_X_MIN, BEV_X_MAX = -X_RANGE_M / 4.0, X_RANGE_M * 3.0 / 4.0
BEV_Y_MIN, BEV_Y_MAX = -Y_RANGE_M / 2.0, Y_RANGE_M / 2.0

# Pixel offsets for converting ego-coordinates to BEV pixel coordinates
BEV_PIXEL_OFFSET_X = GRID_WIDTH_PX / 2.0  # Offset for ego Y-coord to pixel X-coord
BEV_PIXEL_OFFSET_Y = GRID_HEIGHT_PX / 4.0 # Offset for ego X-coord to pixel Y-coord (accounts for forward shift)

# LiDAR point cloud Z-axis filtering and height channel discretization
Z_MIN, Z_MAX = -2.0, 3.8                  # Min/Max Z value (meters) for points to be included in BEV
LIDAR_HEIGHT_CHANNELS = 29                # Number of height slices for LiDAR BEV
LIDAR_SWEEPS = 10                         # Number of past LiDAR sweeps to aggregate
LIDAR_TOTAL_CHANNELS = LIDAR_HEIGHT_CHANNELS * LIDAR_SWEEPS # Total input channels for LiDAR BEV

# --- Map Configuration ---
MAP_CHANNELS = 9 # Number of channels for rasterized map features
# Semantic meaning of each map channel:
# 0: Drivable Area, 1: Lane Boundary (Left), 2: Lane Boundary (Right),
# 3: Pedestrian Crossing, 4: Intersection Area, 5: Bus Lane Area,
# 6: Dashed White Lane Marking, 7: Solid White Lane Marking, 8: Solid Yellow Lane Marking

# --- Intention Configuration ---
NUM_INTENTION_CLASSES = 8                 # Total number of distinct intention classes
INTENTION_HORIZON_SECS = 3.0              # Prediction horizon for intention labels in seconds
INTENTION_HORIZON_STEPS = int(INTENTION_HORIZON_SECS * 10) # Horizon in steps (assuming 10Hz data)

MIN_SPEED_STOPPED = 0.5                   # m/s: Speed threshold to consider a vehicle stopped
MIN_SPEED_MOVING = 1.0                    # m/s: Speed threshold to consider a vehicle actively maneuvering

HEADING_CHANGE_THRESH_TURN = np.radians(20) # Radians: Heading change over horizon for turn classification
HEADING_CHANGE_THRESH_LANE_KEEP = np.radians(5) # Radians: Max heading change for lane keeping

PARKED_MAX_DISP_M = 0.5                   # Meters: Max displacement for a vehicle to be considered parked
KEEP_LANE_MAX_LAT_DIST_FALLBACK = 0.5     # Meters: Max lateral deviation for keep-lane (fallback if no map)

# Mapping from intention string labels to integer class indices
INTENTIONS_MAP = {
    "KEEP_LANE": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2,
    "LEFT_CHANGE_LANE": 3, "RIGHT_CHANGE_LANE": 4,
    "STOPPING_STOPPED": 5, "PARKED": 6, "OTHER": 7
}
INTENTIONS_MAP_REV = {v: k for k, v in INTENTIONS_MAP.items()} # Reverse mapping

# Configuration for handling class imbalance in intention labels during training
DOMINANT_CLASSES_FOR_DOWNSAMPLING = {
    INTENTIONS_MAP["KEEP_LANE"],
    INTENTIONS_MAP["OTHER"],        # 'OTHER' can be frequent depending on heuristic
    INTENTIONS_MAP["PARKED"],
}
INTENTION_DOWNSAMPLE_RATIO = 0.85  # Proportion of dominant class samples to discard

# --- Dataset Configuration ---
# Categories of vehicles to be considered from Argoverse 2 annotations
VEHICLE_CATEGORIES = {
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK", "TRUCK",
    "MOTORCYCLE", "SCHOOL_BUS", "ARTICULATED_BUS", "VEHICULAR_TRAILER",
    "TRUCK_CAB", "BICYCLE", "BICYCLIST", "MOTORCYCLIST"
}

# Note: Paths for data, models, and script-specific hyperparameters (batch size, LR, etc.)
# are typically defined in the respective train/eval scripts or passed as arguments.