import cv2
import numpy as np

import constants

LK_PARAMS = dict(winSize=(31, 31),
                 maxLevel=7,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

FEATURE_PARAMS = dict(qualityLevel=0.3,
                      minDistance=3,
                      blockSize=7)

MAX_OPTICAL_FEATURES = 150


class OpticalFlow:
    """
    Class for storing old positions and calculation optical flow of these positions
    """

    @staticmethod
    def draw(image, serialized_optical_flow) -> np.ndarray:
        """
        Helper function to draw optical flow

        :param image: selected image to draw on
        :param serialized_optical_flow: serialized optical flow
        :return: updated image
        """

        new_positions, old_positions = serialized_optical_flow

        for i, (new, old) in enumerate(zip(new_positions, old_positions)):
            a, b = new.ravel()
            c, d = old.ravel()

            cv2.circle(image, (int(c), int(d)), 3, constants.COLOR_RED, 1)
            cv2.line(img=image,
                     pt1=(int(a), int(b)),
                     pt2=(int(c), int(d)),
                     color=constants.COLOR_RED,
                     thickness=2)

        return image

    def __init__(self, info, tracked_objects_repository):
        """
        :param info: instance of InputInfo
        :param tracked_objects_repository: reference to tracked objects repository
        """

        self._new_positions = []
        self._old_positions = []
        self._previous_image = None
        self._features_to_track = None

        self._tracked_objects_repository = tracked_objects_repository
        self._info = info
        self._doter = np.zeros(shape=(info.height, info.width), dtype=np.uint8)

        for x in range(int(info.width / constants.OPTICAL_FLOW_GRID_DENSITY)):
            for y in range(int(info.height / constants.OPTICAL_FLOW_GRID_DENSITY)):
                self._doter[y * constants.OPTICAL_FLOW_GRID_DENSITY][x * constants.OPTICAL_FLOW_GRID_DENSITY] = 255

    @property
    def tracked_point_count(self):
        """
        :return: number of current optical flow tracked points
        """

        return self._features_to_track.shape[0]

    def update(self, new_frame):
        """
        Updates optical flow and delegates changes to tracked object repository

        :param new_frame: new frame to calculate optical flow on
        """

        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
        mask_for_detection = np.zeros_like(new_frame_gray)

        if self._previous_image is not None:

            for tracked_object in self._tracked_objects_repository.list:
                cv2.circle(img=mask_for_detection,
                           center=tracked_object.center.tuple(),
                           radius=tracked_object.area("outer"),
                           color=constants.COLOR_WHITE_MONO,
                           thickness=constants.FILL)

            if self.tracked_point_count:
                moved_grid, st, err = cv2.calcOpticalFlowPyrLK(prevImg=self._previous_image,
                                                               nextImg=new_frame_gray,
                                                               prevPts=self._features_to_track.astype(np.float32),
                                                               nextPts=None,
                                                               **LK_PARAMS)

                self._new_positions = moved_grid[st == 1]
                self._old_positions = self._features_to_track[st == 1]
            else:
                self._new_positions = []
                self._old_positions = []

        self._previous_image = new_frame_gray
        self._features_to_track = np.zeros(shape=(0, 1, 2), dtype=np.float32)

        if self._tracked_objects_repository.count:
            mask = self._tracked_objects_repository.all_boxes_mask(area_size="inner")
            out_masked = cv2.bitwise_and(self._doter, mask)

            nonzero = cv2.findNonZero(out_masked)

            self._features_to_track = nonzero
            for tracked_object in self._tracked_objects_repository.list:
                tracked_object.update_flow(self._old_positions, self._new_positions)

            try:
                self._features_to_track = np.unique(self._features_to_track, axis=0)
            except np.AxisError:
                self._features_to_track = np.zeros(shape=(0, 1, 2), dtype=np.float32)

    def serialize(self) -> ([], []):
        """
        :return: serialized optical flow (old positions, new positions)
        """

        return self._old_positions, self._new_positions
