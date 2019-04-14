import cv2
import params

from pipeline import ThreadedPipeBlock


class ViolationWriter(ThreadedPipeBlock):

    def __init__(self, info):
        super().__init__(pipe_id=params.VIOLATION_WRITER_ID, work_modes=params.VIOLATION_WRITER_WORKMODES)
        self._info = info

        self._video_writers = []

    def _step(self, seq):
        loader_seq, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=params.OBSERVER_ID)












# class VideoWriter(ThreadedPipeBlock):
#     def __init__(self, info, output_name='output'):
#         super().__init__(pipe_id=params.VIDEO_WRITER_ID)
#
#         self._info = info
#         self._output = cv2.VideoWriter(output_name + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), self._info.fps, (self._info.width, self._info.height))
#
#     def _step(self, seq):
#         loader_seq, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
#         observer_seq, boxes_repository, lights_state = self.receive(pipe_id=params.OBSERVER_ID)
#
#         image = boxes_repository.draw(image)
#         image = self._info.draw_vanishing_points(image)
#
#         # image_with_corridors = self._info.draw_corridors(image)
#         image = self._info.draw_detected_traffic_lights(image)
#         image = self._info.draw_syntetic_traffic_lights(image, lights_state)
#
#         self._output.write(image)
#
#     def _after(self):
#         self._output.release()
