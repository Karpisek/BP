from enum import Enum

import cv2
import tensorflow as tf
import numpy as np

import params
from bbox import Coordinates
from repositories.base.repository import Repository


class Color(Enum):
    RED = 0
    RED_ORANGE = 1
    ORANGE = 2
    GREEN = 3
    NONE = 4


class TrafficLightsRepository(Repository):

    def __init__(self, model, info):
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
        return self.size > 0

    @property
    def size(self):
        return len(self._traffic_lights)

    def state(self, image):
        return self._traffic_lights[0].state_counts(image)

    def select_manually(self, image):
        rectangle = cv2.selectROI("select_traffic_light", image, showCrosshair=True)
        cv2.destroyWindow("select_traffic_light")

        x, y, width, height = rectangle

        top_left = Coordinates(x, y)
        bottom_right = Coordinates(x + width, y + height)

        self.add_traffic_light(top_left=top_left, bottom_right=bottom_right)

    def add_traffic_light(self, top_left, bottom_right):
        new_traffic_light = TrafficLight(top_left=top_left, bottom_right=bottom_right)
        self._traffic_lights.append(new_traffic_light)

    def find(self, image):
        img_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes) = self.sess.run(
            [self.d_boxes, self.d_scores, self.d_classes],
            feed_dict={self.image_tensor: img_expanded})

        for box, score, class_id in list(zip(boxes[0], scores[0], classes[0])):

            if score < 0.5:
                break

            if class_id == 10:
                y_min, x_min, y_max, x_max = box

                top_left = Coordinates(x_min, y_min, info=self._info)
                bottom_right = Coordinates(x_max, y_max, info=self._info)
                self.add_traffic_light(top_left=top_left, bottom_right=bottom_right)
                break

    def draw(self, image):
        for light in self._traffic_lights:
            light.draw_contours(image)

        return image


class TrafficLight:
    def __init__(self, top_left, bottom_right):
        self._top_left = top_left.tuple()
        self._bottom_right = bottom_right.tuple()

        print(self._top_left, self._bottom_right)

    def state_counts(self, image):
        light_roi = image[self._top_left[1]: self._bottom_right[1], self._top_left[0]: self._bottom_right[0]]

        smoothed_light_roi = cv2.medianBlur(light_roi, 5)

        hsv_image = cv2.cvtColor(smoothed_light_roi, cv2.COLOR_BGR2HSV)

        low_red_mask = cv2.inRange(hsv_image, (0, 100, 20), (5, 255, 255))
        up_red_mask = cv2.inRange(hsv_image, (160, 100, 20), (180, 255, 255))
        orange_mask = cv2.inRange(hsv_image, (10, 100, 20), (25, 255, 255))
        green_mask = cv2.inRange(hsv_image, (30, 100, 20), (90, 255, 255))

        red_mask = np.maximum(low_red_mask, up_red_mask)

        red_count = cv2.countNonZero(red_mask)
        orange_count = cv2.countNonZero(orange_mask)
        green_count = cv2.countNonZero(green_mask)

        all_count = red_count + orange_count + green_count

        if all_count < 30:
            return 0, 0, 0
        else:
            return red_count/all_count, orange_count/all_count, green_count/all_count

    def draw_contours(self, image):
        cv2.rectangle(img=image,
                      pt1=self._top_left,
                      pt2=self._bottom_right,
                      thickness=params.DEFAULT_THICKNESS,
                      color=params.COLOR_YELLOW)



