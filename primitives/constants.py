"""
Contains used constants in this project, changing those may lead to errors.
"""

__author__ = "Miroslav Karpisek"
__email__ = "xkarpi05@stud.fit.vutbr.cz"
__date__ = "14.5.2019"

import numpy as np

# pipeline block id's
VIDEO_PLAYER_ID = 0
FRAME_LOADER_ID = 1
DETECTOR_CAR_ID = 2
TRACKER_ID = 3
CALIBRATOR_ID = 4
DETECTOR_LIGHT_ID = 5
LIGHT_FINDER_ID = 6
OBSERVER_ID = 7
TRAFFIC_LIGHT_OBSERVER_ID = 8
VIDEO_WRITER_ID = 9
VIOLATION_WRITER_ID = 10

# traffic violation writer
VIOLATION_WRITER_FREQUENCY = 1
VIOLATION_WRITER_SEQUENCE_LENGTH = 30

# calibrator
CALIBRATOR_FREQUENCY = 50                   # calibrator runs on every N frame
CALIBRATOR_RANSAC_THRESHOLD_RATIO = 0.01    # ransac acceptance of point threshold 0.03 (o.O15)
CALIBRATOR_VP1_TRACK_MINIMUM = 30           # number of minimum trajectories for VP1 detection
CALIBRATOR_VP2_TRACK_MINIMUM = 30           # number of minimum edges for VP2 detection
CALIBRATOR_MIN_LINE_LENGTH = 10             # minimal line length used for voting in VP2 detection
CALIBRATOR_MAX_LINE_GAP = 0                 # max line gap on lines used for voting on VP2 detection
CALIBRATOR_HLP_THRESHOLD = 50               # threshold used on lines which participate VP2 detection
CALIBRATOR_LIFELINE_THICKNESS = 100         # thickness of trajectories printed on mask used for corridor detection

# corridors
CORRIDORS_MINIMUM_LIFELINES = 50            # minimum trajectories before corridors may be found
CORRIDORS_STOP_POINTS_MINIMAL = 5           # minimal points for detecting stop line (low number - just for demonstration)
CORRIDORS_RANSAC_THRESHOLD = 100            # ransac threshold used for stop line detection
CORRIDORS_STOP_LINE_OFFSET = 20             # offset of stop line - assuming that cars are heading the -y derection
CORRIDORS_LINE_SELECTOR_THICKNESS = 3       # thickness used for drawing lines while user selection

# detector
DETECTOR_CAR_FREQUENCY = 3                  # detector runs on every N frame
DETECTOR_CAR_CLASSES_IDS = [2, 3, 4, 6, 8]  # IDs of classes detected by detector used in this project
DETECTOR_MINIMAL_SCORE = 0.1                # minimal score of detected class while care detection is done
DETECTOR_LIGHT_MINIMAL_SCORE = 0.5          # minimal score of traffic light detection
DETECTOR_IMAGE_WIDTH = 640                  # width of image passed to detector

# tracker
TRACKER_OPTICAL_FLOW_FREQUENCY = 1          # tracker runs on every N frame
TRACKER_DISALLOWED = 10000                  # constant representing infinity in munkres assigning algorithm
TRACKER_LIFETIME = 20                       # max lifetime of tracked object after crossing stop line
TRACKER_MINIMAL_SCORE = 0.5                 # minimal score for creating new car instance
TRACKER_MAX_OVERLAP = 0.1                   # max overlap of two tracked objects
TRACKER_TRAJECTORY_MIN_ANGLE = 30           # minimal angle on selecting cars heading to VP1
TRACKER_TRAJECTORY_MAX_ANGLE = 150          # maximal angle used on selecting cars heading to VP1
TRACKER_HISTORY_DIFFERENCE = 20             # minimal y-coordinate difference to actual tracker point to be saved in history

# video player
VIDEO_PLAYER_FREQUENCY = 1                  # video player runs on every N frame
VIDEO_PLAYER_SPEED = 10                     # N milliseconds pause between step iterations

# video writer
VIDEO_WRITER_FREQUENCY = 1                  # video writer runs on every N frame
VIDEO_WRITER_HISTORY = 20                   # number of frames before and after violation to be recorded

# frame loader
FRAME_LOADER_MAX_WIDTH = 1100               # maximal width of images - used for rescale
FRAME_LOADER_MAX_FPS = 20                   # maximal FPS for video writer - otherwise is lowered

# optical flow
OPTICAL_FLOW_GRID_DENSITY = 5               # density of dense optical flow grid used for vehicle tracking

# traffic light
TRAFFIC_LIGHT_CLASS_ID = 10                 # ID of traffic light class
TRAFFIC_LIGHT_DEFAULT_THRESHOLD = 0.3       # relative threshold used for color filtering
TRAFFIC_LIGHT_GREEN_THRESHOLD = 0.4         # relative threshold for green color
TRAFFIC_LIGHT_OBSERVER_FREQUENCY = 1        # traffic light observer runs on every N frame

# observer
OBSERVER_FREQUENCY = 1                      # observer runs on every N frame
OBSERVER_BOX_THICKNESS = 3                  # thickness of box printed by observer

# colors - helpers for vizualization
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_WHITE_MONO = 255
COLOR_BLACK = (0, 0, 0)
COLOR_LIFELINE = (10, 10, 10)
COLOR_ORANGE = (66, 212, 244)
COLOR_GRAY = (50, 50, 50)
COLOR_YELLOW = (32, 244, 66)
OPTICAL_FLOW_COLOR = (200, 200, 50)
COLOR_AREA = (255, 255, 204)

FILL = -1                                   # OpenCV constant for fill
DEFAULT_THICKNESS = 1                       # default thickness

# used filenames
STATISTICS_LOG_FILENAME = "statistics.json"
CALIBRATION_FILENAME = "calibration.json"
ANNOTATIONS_FILENAME = "annotations.json"

# another constants
UINT_MIN = -2147483648
UINT_MAX = 2147483647
RANDOM_COLOR = [(np.random.randint(10, 100), np.random.randint(10, 100), np.random.randint(10, 100)) for _ in range(100)]
COLOR_VANISHING_DIRECTIONS = [COLOR_GREEN, COLOR_BLUE, COLOR_RED]
AREA_THICKNESS = 3

