import tensorflow as tf
import numpy as np

from queue import Queue
from threading import Thread

from bbox import Box2D
from bbox.coordinates import Coordinates
from bbox.size import ObjectSize
from pipeline import PipeBlock


class Detector(PipeBlock):

    def __init__(self, model, detection_area, output=None):

        super().__init__(output)

        self._input = [Queue(20)]

        self.detection_area = detection_area

        self.detection_graph = tf.Graph()

        self._thread = Thread(target=self._run, args=(model,))
        self._thread.daemon = True
        self._thread.start()

    def _run(self, model):

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
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

        while True:
            seq, image = self.next()

            img_expanded = np.expand_dims(image, axis=0)

            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})

            packet_boxes = self.parse_boxes(boxes, scores)

            self.send_to_all(packet_boxes, pipe=0)

    def parse_boxes(self, boxes, scores):

        final_boxes = []
        for _, pair in enumerate(zip(boxes[0], scores[0])):
            box, score = pair
            if score < Box2D.MINIMAL_SCORE_CORRECTION:
                break

            center, size = Detector.convert_box_to_centroid_object(box)
            new_box = (center, size, score)

            if self.detection_area.contains(center, relative=True):
                final_boxes.append(new_box)

        return final_boxes

    @staticmethod
    def convert_box_to_centroid_object(box):
        y_min, x_min, y_max, x_max = box

        x = (x_min + x_max)/2
        y = (y_min + y_max)/2

        center = Coordinates(x, y)

        width = x_max - x_min
        height = y_max - y_min

        size = ObjectSize(width, height)

        return center, size
