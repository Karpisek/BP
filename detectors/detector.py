import tensorflow as tf
import numpy as np
import params

from bbox import ObjectSize, Coordinates
from pipeline import ThreadedPipeBlock


class Detector(ThreadedPipeBlock):
    """
    PipeBlock which uses model of neural network for object detection.
    Any type of trained neural network for detection could be passed.
    Code for running tensorflow model inspired by: https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-5-saving-and-deploying-a-model-8d51f56dbcf1
    """

    def _mode_changed(self, new_mode):
        pass

    def __init__(self, info, model, output=None, detector_type_id=params.DETECTOR_CAR_ID, block=True, max_steps=np.inf):
        """
        :param info: instance of InputInfo
        :param model: path to trained object detection model
        :param output: list of PipeBlock output instances
        :param detector_type_id: unique ID used for communication between PipeBlocks
        :param max_steps: maximum number of steps
        """

        super().__init__(info=info, pipe_id=detector_type_id, output=output, max_steps=max_steps)
        self.detection_graph = tf.Graph()
        self._block = block

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
        """
        On each steps, if sequence number compare with detector frequency is correct new detection is done.
        Detected boxes are converted from relative coordinate to real coordinate using InputInfo.
        If detected box is outside specified area it is thrown away.

        :param seq:
        :return:
        """
        seq, image = self.receive(params.FRAME_LOADER_ID)
        img_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, _) = self.sess.run(
            [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
            feed_dict={self.image_tensor: img_expanded})

        packet_boxes = self.parse_boxes(boxes[0], scores[0], classes[0])

        if len(self._output):
            self.send(packet_boxes, pipe_id=list(self._output.keys())[0], block=self._block)

    def parse_boxes(self, boxes, scores, classes) -> [(Coordinates, ObjectSize, float)]:
        """
        Filters away boxes with low score or classes which are not specified. And boxes which are not located
        in specified area as well as boxes which are to wide.
        Converts boxes from relative to real coordinates and makes them be specified by centroid object.

        :param boxes: bounding boxes
        :param scores: scores of bounding boxes
        :param classes: classes of bounding boxes
        :return: list of filtered boxes
        """

        final_boxes = []
        for _, squad in enumerate(zip(boxes, scores, classes)):
            box, score, class_id = squad

            if score < params.DETECTOR_MINIMAL_SCORE:
                break

            if class_id not in params.DETECTOR_CAR_CLASSES_IDS:
                continue

            center, size = self.convert_box_to_centroid_object(box)

            if size.square_size < 0.1 and center in self._info.update_area:
                new_box = (center, size, score, class_id)
                final_boxes.append(new_box)

        return final_boxes

    def convert_box_to_centroid_object(self, box) -> (Coordinates, ObjectSize):
        """
        Converts box which is defined by top left and bottom right anchor into box defined by middle point
        and width and height

        :param box: bounding box to convert
        :return: converted bounding box
        """

        y_min, x_min, y_max, x_max = box

        x = (x_min + x_max)/2
        y = (y_min + y_max)/2

        center = Coordinates(x, y, info=self._info)

        width = x_max - x_min
        height = y_max - y_min

        size = ObjectSize(width, height, info=self._info)

        return center, size
