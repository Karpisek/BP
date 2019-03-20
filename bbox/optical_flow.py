import cv2
import numpy as np

from bbox import Box2D

LK_PARAMS = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

FEATURE_PARAMS = dict(maxCorners=150,
                      qualityLevel=0.3,
                      minDistance=3,
                      blockSize=7)


class OpticalFlow:
    color = np.random.randint(0, 255, (10000, 3))

    def __init__(self):
        self._new_positions = []
        self._old_positions = []
        self._previous_image = None
        self._features_to_track = None

    @staticmethod
    def draw(image, serialized_optical_flow) -> np.ndarray:
        mask = np.zeros_like(image)
        new_positions, old_positions = serialized_optical_flow
        for i, (new, old) in enumerate(zip(new_positions, old_positions)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (a, b), (c, d), OpticalFlow.color[i].tolist(), 1)

        return mask

    def update(self, new_frame):

        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        mask_for_detection = np.zeros_like(new_frame_gray)

        if self._previous_image is not None:

            for box in Box2D.boxes:
                cv2.circle(mask_for_detection, box.center.tuple(), box.area_of_interest(), 255, -1)

            if self._features_to_track is not None:
                moved_grid, st, err = cv2.calcOpticalFlowPyrLK(self._previous_image, new_frame_gray,
                                                               self._features_to_track, None, **LK_PARAMS)

                self._new_positions = moved_grid[st == 1]
                self._old_positions = self._features_to_track[st == 1]
            else:
                self._new_positions = []
                self._old_positions = []

        self._previous_image = new_frame_gray
        self._features_to_track = cv2.goodFeaturesToTrack(self._previous_image, mask=mask_for_detection, **FEATURE_PARAMS)

        for box in Box2D.boxes:
            box.update_flow(self._old_positions, self._new_positions)

    def serialize(self) -> ([], []):
        return self._old_positions, self._new_positions
