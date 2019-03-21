import tensorflow as tf
import numpy as np

from bbox import Box2D, ObjectSize, Coordinates
from params import FRAME_LOADER_ID, TRACKER_ID, DETECTOR_ID
from pipeline import ThreadedPipeBlock


class Detector(ThreadedPipeBlock):

    def __init__(self, model, detection_area, output=None):

        super().__init__(pipe_id=DETECTOR_ID, output=output)

        self.detection_area = detection_area

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
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def _step(self, seq):
        seq, image = self.receive(FRAME_LOADER_ID)

        img_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
            feed_dict={self.image_tensor: img_expanded})

        packet_boxes = self.parse_boxes(boxes, scores)

        self.send(packet_boxes, pipe_id=TRACKER_ID)

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
