import time

import cv2
import numpy as np

import params
from bbox import Coordinates
from pipeline import ThreadedPipeBlock


class TrafficLightsFinder(ThreadedPipeBlock):

    def __init__(self, info):
        super().__init__(pipe_id=params.LIGHT_FINDER_ID)

        self._finder_mask = np.zeros(shape=(info.height, info.width), dtype=np.uint8)
        self._helper_mask = np.full(shape=(info.height, info.width), dtype=np.uint8, fill_value=244)

        self._info = info

    def _step(self, seq):
        detected_objects = self.receive(pipe_id=params.DETECTOR_LIGHT_ID)

        for detected_object in detected_objects:
            self._draw_detected_object(*detected_object)

        if np.amax(self._finder_mask) == 255:
            all_finder_mask = np.copy(self._finder_mask)
            all_finder_mask[all_finder_mask > 0] = 255
            all_finder_mask = cv2.bitwise_not(all_finder_mask)

            self._finder_mask[self._finder_mask < 200] = 0

            contours, hierarchy = cv2.findContours(self._finder_mask, 1, 2)

            seed_points = []
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                x, y = rect[0]
                w, h = rect[1]

                center = int(x + w/2), int(y + h/2)
                print(center)
                seed_points.append(center)

            tabula_rasa = np.zeros_like(all_finder_mask)
            all_finder_mask = np.pad(all_finder_mask, 1, 'constant', constant_values=255)

            for point in seed_points:
                cv2.floodFill(image=tabula_rasa,
                              mask=all_finder_mask,
                              seedPoint=point,
                              newVal=50)

            print("ahoj")
            cv2.imwrite(f"10.jpg", tabula_rasa)






            cv2.imwrite(f"2.jpg", self._finder_mask)

            exit()

    def _draw_detected_object(self, coordinates, size, confident_score) -> None:
        print(confident_score)
        if confident_score < params.TRAFFIC_LIGHT_MINIMAL_SCORE:
            return

        point1 = Coordinates(x=coordinates.x - size.width / 2,
                             y=coordinates.y - size.height / 2)

        point2 = Coordinates(x=coordinates.x + size.width / 2,
                             y=coordinates.y + size.height / 2)

        color = int(params.TRAFFIC_LIGHT_FINDER_DEFAULT_VALUE * confident_score)

        helper_mask = np.copy(self._finder_mask)
        cv2.rectangle(img=helper_mask,
                      pt1=point1.tuple(),
                      pt2=point2.tuple(),
                      color=color,
                      thickness=params.FILL)

        self._finder_mask = cv2.add(helper_mask, self._finder_mask)



