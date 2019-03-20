import cv2
import numpy as np

from bbox import Box2D

LK_PARAMS = dict(winSize=(41, 41),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

FEATURE_PARAMS = dict(qualityLevel=0.3,
                      minDistance=3,
                      blockSize=7)

MAX_OPTICAL_FEATURES = 150

GRID_DENSITY = 7


class OpticalFlow:
    OPTICAL_FLOW_COLOR = (200, 200, 50)

    def __init__(self):
        self._new_positions = []
        self._old_positions = []
        self._previous_image = None
        self._features_to_track = None

        self._doter = None

    @staticmethod
    def draw(image, serialized_optical_flow) -> np.ndarray:
        mask = np.zeros_like(image)
        new_positions, old_positions = serialized_optical_flow
        for i, (new, old) in enumerate(zip(new_positions, old_positions)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (int(a), int(b)), (int(c), int(d)), OpticalFlow.OPTICAL_FLOW_COLOR, 1)

        return mask

    @property
    def tracked_point_count(self):
        return self._features_to_track.shape[0]

    def init(self, image):
        height, width, _ = image.shape

        self._doter = np.zeros(shape=(height, width), dtype=np.uint8)

        for x in range(int(width / GRID_DENSITY)):
            for y in range(int(height / GRID_DENSITY)):
                self._doter[y * GRID_DENSITY][x * GRID_DENSITY] = 255

    def update(self, new_frame):

        if self._doter is None:
            self.init(new_frame)

        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        mask_for_detection = np.zeros_like(new_frame_gray)

        if self._previous_image is not None:

            for box in Box2D.boxes:
                cv2.circle(mask_for_detection, box.center.tuple(), box.area_of_interest(), 255, -1)

            if self.tracked_point_count:
                moved_grid, st, err = cv2.calcOpticalFlowPyrLK(self._previous_image, new_frame_gray,
                                                               self._features_to_track.astype(np.float32), None, **LK_PARAMS)

                self._new_positions = moved_grid[st == 1]
                self._old_positions = self._features_to_track[st == 1]
            else:
                self._new_positions = []
                self._old_positions = []

        self._previous_image = new_frame_gray
        self._features_to_track = np.zeros(shape=(0, 1, 2), dtype=np.float32)
        if len(Box2D.boxes):
            for box in Box2D.boxes:

                mask = box.mask(new_frame_gray)
                out_masked = cv2.bitwise_and(self._doter, mask)

                nonzero = cv2.findNonZero(out_masked)
                if nonzero is not None:
                    self._features_to_track = np.concatenate((self._features_to_track, nonzero))

            for box in Box2D.boxes:
                box.update_flow(self._old_positions, self._new_positions)

    def serialize(self) -> ([], []):
        return self._old_positions, self._new_positions
