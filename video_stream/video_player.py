"""
VideoPlayer class definition
"""

__author__ = "Miroslav Karpisek"
__email__ = "xkarpi05@stud.fit.vutbr.cz"
__date__ = "14.5.2019"

import cv2
import constants

from bbox import TrackedObject
from pipeline import PipeBlock
from pipeline.base.pipeline import Mode


class UserEndException(Exception):
    pass


class VideoPlayer(PipeBlock):
    """
    PipeBlock subclass used for replaying current progress of calibration and red light violation detection. Not to be
    meant in final product just for displaying the method used in this project.
    Creates window displaying frame by frame. Allows user interaction by keystrokes.
    """

    def __init__(self, info, print_fps, output):
        """
        :param info: instance of InputInfo class containing all informations about examined video.
        :param print_fps: if approximate fps should be printed on console
        :param output: list of output PipeBlocks. Used for delegation of user interaction between PipeBlocks.
        """

        super().__init__(info=info, pipe_id=constants.VIDEO_PLAYER_ID, print_fps=print_fps, output=output)

        self._detector = None
        self._loader = None
        self._tracker = None

        self._draw_trajectories = False
        self._draw_corridors = False
        self._draw_vanishing_points = False
        self._draw_lights = False
        self._draw_statistics = False
        self._draw_boxes = True
        self._draw_area = False
        self._draw_flow = False

    def _mode_changed(self, new_mode):
        super()._mode_changed(new_mode)

    def _before(self):
        pass

    def _step(self, seq):
        """
        Receives new frame from FrameLoader and all informations about current scene from Observer (car boxes
        and traffic light info)

        :param seq: current sequence number
        """

        loader_seq, image = self.receive(pipe_id=constants.FRAME_LOADER_ID)
        observer_seq, boxes_repository, lights_state = self.receive(pipe_id=constants.OBSERVER_ID)
        sequence_number, serialized_tracked_objects, tracked_object_lifelines, flows = self.receive(pipe_id=constants.TRACKER_ID)

        image = self._assemble_frame(image, boxes_repository, lights_state, flows)

        cv2.imshow("image", image)

        self._parse_user_interaction(image)
        self.send(None, constants.VIOLATION_WRITER_ID)

    def _parse_user_interaction(self, image):
        """
        Sets the time between frames using cv2.waitKey() and provides behavior depending on user interaction.
        On supported Keystroke inside flags are being updated.
        """

        key = cv2.waitKey(constants.VIDEO_PLAYER_SPEED)

        if key & 0xFF == ord("q"):
            raise EOFError

        # print
        if key & 0xFF == ord("p"):
            cv2.imwrite("/Users/miro/Desktop/bp_photos/printP.png", image)

        # default
        if key & 0xFF == ord("d"):
            self._draw_trajectories = False
            self._draw_corridors = False
            self._draw_boxes = True
            self._draw_vanishing_points = False
            self._draw_statistics = True
            self._draw_area = False

        # trajectories
        if key & 0xFF == ord("t"):
            self._draw_trajectories = not self._draw_trajectories

        # corridors
        if key & 0xFF == ord("c"):
            self._draw_corridors = not self._draw_corridors

        # vanishing points
        if key & 0xFF == ord("v"):
            self._draw_vanishing_points = not self._draw_vanishing_points

        # lights
        if key & 0xFF == ord("l"):
            self._draw_lights = not self._draw_lights

        #  statistics
        if key & 0xFF == ord("s"):
            self._draw_statistics = not self._draw_statistics

        #  statistics
        if key & 0xFF == ord("b"):
            self._draw_boxes = not self._draw_boxes

        #  area
        if key & 0xFF == ord("a"):
            self._draw_area = not self._draw_area

        #  flow
        if key & 0xFF == ord("f"):
            self._draw_flow = not self._draw_flow

    def _after(self):
        """
        Destroys all windows after computation is done.
        Delegates information about closing to ViolationWriter (in case it was caused by user interaction)
        """

        cv2.destroyAllWindows()
        self._update_mode(Mode.SIGNAL)
        self.send(EOFError, constants.VIOLATION_WRITER_ID)

    def _assemble_frame(self, image, boxes_repository, lights_state, flows):
        """
        Depending on inside flags draws additional information to current frame.
        inside flags can be set by user interaction.

        :param image: selected image
        :param boxes_repository: instance of repository of car bounding boxes
        :param lights_state: current light state
        :return: updated image
        """

        if self._draw_trajectories:
            image = boxes_repository.draw_trajectories(image)

        if self._draw_boxes:
            image = boxes_repository.draw_boxes(image)

        if self._draw_lights:
            image = self._info.draw_detected_traffic_lights(image)
            self._info.draw_synthetic_traffic_lights(image, lights_state)

        if self._draw_corridors:
            image = self._info.draw_corridors(image)

        if self._draw_statistics:
            image = boxes_repository.draw_statistics(image, self._info)

        if self._draw_vanishing_points:
            image = self._info.draw_vanishing_points(image)

        if self._draw_area:
            image = self._info.update_area.draw(image)

        if self._draw_flow:
            image = TrackedObject.draw_flow(image, flows)

        return image
