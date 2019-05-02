import cv2

from helper_scripts.base_annotator import BaseAnnotator


class CarAnnotator(BaseAnnotator):
    def _step(self, seq, image):
        cv2.imshow(self.video_info.filename, image)
        k = cv2.waitKey(100)

        if k == 32:
            while True:
                k = cv2.waitKey(1)

                if k == 32:
                    break

                elif k == 99:
                    self._add_annotation("cars", seq)

        elif k == 113:
            raise EOFError

        elif k == 99:
            self._add_annotation("cars", seq)

    def _before(self):
        self.annotations["cars"] = []


annotator = CarAnnotator()
annotator.run()
