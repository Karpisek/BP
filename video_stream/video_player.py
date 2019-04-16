import cv2
import params

from pipeline import PipeBlock


class UserEndException(Exception):
    pass


class VideoPlayer(PipeBlock):
    def __init__(self, info, print_fps):
        super().__init__(pipe_id=params.VIDEO_PLAYER_ID, print_fps=print_fps)

        self._detector = None
        self._loader = None
        self._tracker = None

        self._info = info

    def _mode_changed(self, new_mode):
        pass

    def _before(self):
        pass

    def _step(self, seq):

        loader_seq, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=params.OBSERVER_ID)

        image = boxes_repository.draw(image)

        # image = self._info.draw_vanishing_points(image)

        image = self._info.draw_corridors(image)
        image = self._info.draw_detected_traffic_lights(image)

        self._info.draw_syntetic_traffic_lights(image, lights_state)

        cv2.imshow("image", image)

        key = cv2.waitKey(params.VIDEO_PLAYER_SPEED)

        #  commands
        if key & 0xFF == ord("q"):
            raise EOFError

    def _after(self):
        cv2.destroyAllWindows()
