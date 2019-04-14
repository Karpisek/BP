import cv2
import params

from pipeline import PipeBlock


class VideoWriter(PipeBlock):
    def __init__(self, info):
        super().__init__(pipe_id=params.VIDEO_PLAYER_ID)

        self._info = info
        self._output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self._info.fps, (self._info.width, self._info.height))

    def start(self):

        frame_counter = 0

        seq, image = self.receive(params.FRAME_LOADER_ID)
        frame_counter += 1

        while True:
            observer_seq, boxes_repository = self.receive(pipe_id=params.OBSERVER_ID)

            image = boxes_repository.draw(image)
            image = self._info.draw_vanishing_points(image)

            # image_with_corridors = self._info.draw_corridors(image)
            image = self._info.draw_detected_traffic_lights(image)
            image = self._info.draw_syntetic_traffic_lights(image)

            self._output.write(image)

        self._output.release()
