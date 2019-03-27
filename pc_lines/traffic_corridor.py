import cv2
import numpy as np

import params


class TrafficCorridorRepository:
    def __init__(self, info):
        self._ready = False
        self._info = info
        self._corridors = {}
        self._corridors_count = 0
        self._corridor_mask = np.zeros(shape=(info.height, info.width), dtype=np.uint8)

    @property
    def corridor_mask(self):
        return self._corridor_mask

    def get_mask(self) -> np.ndarray:

        image = np.zeros(shape=(self._info.height, self._info.width, 3),
                         dtype=np.uint8)

        if self._ready:
            for key, corridor in self._corridors.items():
                corridor.draw_corridor(image, self._info, color=params.RANDOM_COLOR[key])

        return image

    def create_new_corridor(self, left_point, right_point):
        self._corridors_count += 1
        new_corridor = TrafficCorridor(index=self._corridors_count,
                                       left_point=left_point,
                                       right_point=right_point)

        self._corridors[new_corridor.id] = new_corridor

        new_corridor.draw_corridor(self._corridor_mask, self._info)

    def find_corridors(self, lifelines_mask):

        # extract left-bot-right 1px border around lifeline image and make "1D array"
        left_edge = lifelines_mask[:, :1]
        bottom_edge = lifelines_mask[-1:, :].transpose(1, 0, 2)
        right_edge = lifelines_mask[:, -1:]
        frame_edge = np.concatenate((left_edge, bottom_edge, right_edge))

        # greyscale image and reduce noise by multiple blur and threshold
        edge_grey = cv2.cvtColor(frame_edge, cv2.COLOR_RGB2GRAY)

        edge_grey_blured = cv2.blur(edge_grey, (1, 21))
        _, threshold = cv2.threshold(edge_grey_blured, 50, 255, cv2.THRESH_BINARY)

        edge_grey_blured = cv2.blur(threshold, (1, 31))
        _, edge_grey_sharp = cv2.threshold(edge_grey_blured, 10, 255, cv2.THRESH_BINARY)

        height, width = edge_grey_blured.shape

        edge_grey_canny = cv2.Canny(edge_grey_sharp, 50, 150)

        coordinates = []

        # border image with 0
        edge_grey_canny[0][0] = 0
        edge_grey_canny[-1][0] = 0

        for i in range(height):
            if edge_grey_canny[i][0] == 255:
                coordinates.append(i)

        points = np.array(coordinates).reshape(-1, 2)

        for index, point in enumerate(points):
            try:
                half_diff = int((points[index + 1][0] - point[1]) / 2)
            except IndexError:
                break

            point[1] += half_diff
            points[index + 1][0] -= half_diff

        for point in points:
            points = []

            for coordinate in list(point):
                if coordinate < self._info.height:
                    points.append((0, coordinate))
                elif coordinate < self._info.height + self._info.width:
                    points.append((coordinate - self._info.height, self._info.height - 1))
                else:
                    points.append((self._info.width - 1, self._info.width + 2 * self._info.height - coordinate))

            self.create_new_corridor(tuple(points[0]), tuple(points[1]))

        cv2.imwrite("mask.jpg", self.get_mask())
        self._ready = True


class TrafficCorridor:
    def __init__(self, index, left_point, right_point):
        self._id = index
        self.left_point = left_point
        self.right_point = right_point

        print(left_point, right_point)

    @property
    def id(self):
        return self._id

    def draw_corridor(self, image, info, color=None):
        mask = np.zeros(shape=(info.height, info.width),
                        dtype=np.uint8)

        if color is None:
            color = self.id

        cv2.line(image, self.left_point, info.vanishing_points[0].point, color)
        cv2.line(mask, self.left_point, info.vanishing_points[0].point, color)

        cv2.line(image, self.right_point, info.vanishing_points[0].point, color)
        cv2.line(mask, self.right_point, info.vanishing_points[0].point, color)

        middle_point = int((self.left_point[0] + self.right_point[0]) / 2), int((self.left_point[1] + self.right_point[1]) / 2)

        mask_with_border = np.pad(mask, 1, 'constant', constant_values=255)
        cv2.floodFill(image, mask_with_border, middle_point, color)

    def __str__(self):
        return f"coridor: id [{self.id}], coord ({self.left_point}, {self.right_point})"
