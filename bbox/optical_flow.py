import cv2
import numpy as np

import params
from bbox import TrackedObject

LK_PARAMS = dict(winSize=(31, 31),
                 maxLevel=7,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

FEATURE_PARAMS = dict(qualityLevel=0.3,
                      minDistance=3,
                      blockSize=7)

MAX_OPTICAL_FEATURES = 150


class OpticalFlow:
    OPTICAL_FLOW_COLOR = (200, 200, 50)

    def __init__(self, info):
        self._new_positions = []
        self._old_positions = []
        self._previous_image = None
        self._features_to_track = None

        self._info = info
        self._doter = np.zeros(shape=(info.height, info.width), dtype=np.uint8)

        for x in range(int(info.width / params.OPTICAL_FLOW_GRID_DENSITY)):
            for y in range(int(info.height / params.OPTICAL_FLOW_GRID_DENSITY)):
                self._doter[y * params.OPTICAL_FLOW_GRID_DENSITY][x * params.OPTICAL_FLOW_GRID_DENSITY] = 255

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

    def update(self, new_frame):

        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        mask_for_detection = np.zeros_like(new_frame_gray)

        if self._previous_image is not None:

            for box in TrackedObject.boxes:
                cv2.circle(mask_for_detection, box.center.tuple(), box.area("outer"), 255, -1)

            if self.tracked_point_count:
                moved_grid, st, err = cv2.calcOpticalFlowPyrLK(self._previous_image,
                                                               new_frame_gray,
                                                               self._features_to_track.astype(np.float32),
                                                               None,
                                                               **LK_PARAMS)

                self._new_positions = moved_grid[st == 1]
                self._old_positions = self._features_to_track[st == 1]
            else:
                self._new_positions = []
                self._old_positions = []

        self._previous_image = new_frame_gray
        self._features_to_track = np.zeros(shape=(0, 1, 2), dtype=np.float32)

        if len(TrackedObject.boxes):
            mask = TrackedObject.all_boxes_mask(self._info, area_size="inner")
            out_masked = cv2.bitwise_and(self._doter, mask)

            nonzero = cv2.findNonZero(out_masked)

            self._features_to_track = nonzero
            for box in TrackedObject.boxes:

                box.update_flow(self._old_positions, self._new_positions)

            try:
                self._features_to_track = np.unique(self._features_to_track, axis=0)
            except np.AxisError:
                self._features_to_track = np.zeros(shape=(0, 1, 2), dtype=np.float32)

    def serialize(self) -> ([], []):
        return self._old_positions, self._new_positions
