import shutil

import cv2
import os
import params

from pipeline import ThreadedPipeBlock


class ViolationWriter(ThreadedPipeBlock):

    def _mode_changed(self, new_mode):
        pass

    def __init__(self, info):
        super().__init__(pipe_id=params.VIOLATION_WRITER_ID, work_modes=params.VIOLATION_WRITER_WORKMODES, deamon=False)
        self._info = info
        self._video_writers = []
        self._last_boxes_repository = None

    def _before(self):
        path = f"{os.getcwd()}/{self._info.filename}"

        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)

    def _step(self, seq):
        self.receive(pipe_id=params.VIDEO_PLAYER_ID)

        loader_seq, image = self.receive(pipe_id=params.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=params.OBSERVER_ID)
        self._last_boxes_repository = boxes_repository

    def _after(self):
        if self._last_boxes_repository is not None:
            path = f"{os.getcwd()}/{self._info.filename}/statistics.log"

            with open(path, "w") as file:
                self._last_boxes_repository.write_statistics(file)


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
