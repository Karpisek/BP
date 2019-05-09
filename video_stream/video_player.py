import cv2
import numpy as np

import params

from pipeline import PipeBlock
from pipeline.base.pipeline import Mode


class UserEndException(Exception):
    pass


class VideoPlayer(PipeBlock):
    def __init__(self, info, print_fps, output):
        super().__init__(info=info, pipe_id=params.VIDEO_PLAYER_ID, print_fps=print_fps, output=output)

        self._detector = None
        self._loader = None
        self._tracker = None

    def _mode_changed(self, new_mode):
        super()._mode_changed(new_mode)

    def _before(self):
        pass

    def _step(self, seq):

        loader_seq, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=params.OBSERVER_ID)

        image_copy = np.copy(image)

        image = boxes_repository.draw(image, draw_trajectories=False)
        #
        # if self.mode == Mode.CALIBRATION:
        #     # image_copy = self._info.draw_corridors(image_copy)
        image_copy = self._info.draw_vanishing_points(image_copy)
        image_copy = self._info.draw_detected_traffic_lights(image_copy)
        #
        # elif self.mode == Mode.DETECTION:
        #     pass
        image_copy = self._info.draw_corridors(image_copy)

        self._info.draw_syntetic_traffic_lights(image, lights_state)

        if self._mode == Mode.DETECTION:
            image = boxes_repository.draw_statistics(image, self._info)

        cv2.imshow("image", image)
        cv2.imshow("detected segments", image_copy)

        key = cv2.waitKey(params.VIDEO_PLAYER_SPEED)

        #  commands
        if key & 0xFF == ord("q"):
            raise EOFError

        if key & 0xFF == ord("p"):
            cv2.imwrite("/Users/miro/Desktop/bp_photos/corridors.png", image_copy)

        else:
            self.send(None, params.VIOLATION_WRITER_ID)

    def _after(self):
        cv2.destroyAllWindows()
        self._update_mode(Mode.SIGNAL)
        self.send(EOFError, params.VIOLATION_WRITER_ID)
