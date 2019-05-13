from enum import IntEnum

import cv2
import tensorflow as tf
import numpy as np

import constants
from bbox import Coordinates


class Color(IntEnum):
    ORANGE = 0
    RED = 1
    RED_ORANGE = 2
    GREEN = 3
    NONE = 4


class TrafficLightsRepository:
    """
    Repository for storing detected lights (only one light supported for this moment)
    Allows creating and storing these lights.
    Allows detection of traffic light using given model of neural network capable of object detection.
    """

    def __init__(self, model, info):
        """
        :param model: path to neural network used for traffic light detection
        :param info: instance of InputInfo
        """

        self._model = model
        self._info = info
        self._traffic_lights = []

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=self.detection_graph)

    @property
    def ready(self) -> bool:
        """
        :return: if any traffic light was found
        """

        return self.size > 0

    @property
    def size(self):
        """
        :return: number of detected traffic lights
        """

        return len(self._traffic_lights)

    def state(self, current_frame, previous_frame):
        """
        computes actual state of traffic light (only first detected traffic light is used for this moment).
        Uses two images to reduce error.

        :param current_frame: current frame
        :param previous_frame: previous frame
        :return: detected traffic light state
        """

        best_value = 0
        best_state = None

        for light in self._traffic_lights:
            value, red, orange, green = light.state_counts(current_frame, previous_frame)

            if value > best_value:
                best_state = red, orange, green

        if best_state is None:
            return 0, 0, 0
        else:
            return best_state

    def select_manually(self, image):
        """
        Allows user to select traffic light position manualy

        :param image: image to select traffic light on
        """

        rectangle = cv2.selectROI("select_traffic_light", image, showCrosshair=True)
        cv2.destroyWindow("select_traffic_light")

        x, y, width, height = rectangle

        top_left = Coordinates(x, y)
        bottom_right = Coordinates(x + width, y + height)

        self.add_traffic_light(top_left=top_left, bottom_right=bottom_right)

    def add_traffic_light(self, top_left, bottom_right):
        """
        Adds traffic light to storage

        :param top_left: top left anchor of traffic light
        :param bottom_right: bottom right anchor of traffic light
        """

        new_traffic_light = TrafficLight(top_left=top_left, bottom_right=bottom_right)
        self._traffic_lights.append(new_traffic_light)

    def find(self, image):
        """
        Uses neural net for finding traffic light on given image.
        Only the best traffic light detection is added to storage.
        Code inspired by: https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-5-saving-and-deploying-a-model-8d51f56dbcf1

        :param image: selected image
        """

        img_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes) = self.sess.run(
            [self.d_boxes, self.d_scores, self.d_classes],
            feed_dict={self.image_tensor: img_expanded})

        for box, score, class_id in list(zip(boxes[0], scores[0], classes[0])):

            if score < constants.DETECTOR_LIGHT_MINIMAL_SCORE:
                break

            if class_id == constants.TRAFFIC_LIGHT_CLASS_ID:
                y_min, x_min, y_max, x_max = box

                top_left = Coordinates(x_min, y_min, info=self._info)
                bottom_right = Coordinates(x_max, y_max, info=self._info)

                self.add_traffic_light(top_left=top_left, bottom_right=bottom_right)

    def draw(self, image):
        """
        Helper function for drawing detected traffic light position

        :param image: selected image to draw on
        :return: updated image
        """
        for light in self._traffic_lights:
            light.draw_contours(image)

        return image

    def serialize(self):
        return {"traffic light": self._traffic_lights[0].serialize()}


class TrafficLight:
    """
    Represents detected traffic light. Allows to detect current state of this light.
    """

    def __init__(self, top_left, bottom_right):
        """
        :param top_left: top left anchor of detected traffic light
        :param bottom_right: bottom right anchor of detected traffic light
        """
        self._top_left = top_left.tuple()
        self._bottom_right = bottom_right.tuple()

    def state_counts(self, current_frame, previous_frame):
        """
        Returns relative counts of colors in every part of specified color spectre.
        Uses HSV color model for filtering colors.
        For error correction uses combination of current and previous frame

        :param current_frame: current fram
        :param previous_frame: previous frame
        :return:
        """

        current_light = current_frame[self._top_left[1]: self._bottom_right[1], self._top_left[0]: self._bottom_right[0]]
        previous_light = previous_frame[self._top_left[1]: self._bottom_right[1], self._top_left[0]: self._bottom_right[0]]

        smoothed_current_light = cv2.blur(current_light, (3, 3))
        smoothed_previous_light = cv2.blur(previous_light, (3, 3))

        combined_image = cv2.max(smoothed_current_light, smoothed_previous_light)

        hsv_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2HSV)

        low_red_mask = cv2.inRange(hsv_image, (0, 100, 10), (5, 255, 255))
        up_red_mask = cv2.inRange(hsv_image, (160, 100, 10), (180, 255, 255))
        orange_mask = cv2.inRange(hsv_image, (5, 100, 10), (25, 255, 255))
        green_mask = cv2.inRange(hsv_image, (30, 100, 10), (95, 255, 255))

        red_mask = np.maximum(low_red_mask, up_red_mask)

        red_count = cv2.countNonZero(red_mask)
        orange_count = cv2.countNonZero(orange_mask)
        green_count = cv2.countNonZero(green_mask)

        all_count = red_count + orange_count + green_count

        if all_count < 10:  # 30
            return 0, 0, 0, 0
        else:
            return all_count, red_count/all_count, orange_count/all_count, green_count/all_count

    def draw_contours(self, image):
        """
        Helper function for contoure drawing of detected traffic light

        :param image: selected image to draw on
        """

        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=self._bottom_right,
                      thickness=constants.DEFAULT_THICKNESS,
                      color=constants.COLOR_YELLOW)

    def serialize(self):
        """
        :return: serialized traffic light
        """

        return {"top left": list(self._top_left), "bottom right": list(self._bottom_right)}



