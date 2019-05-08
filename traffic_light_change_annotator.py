import cv2

from base_annotator import BaseAnnotator

state = 0


class LightAnnotator(BaseAnnotator):
    def _step(self, seq, image):
        cv2.imshow(self.video_info.filename, image)
        k = cv2.waitKey(30)

        if k == 32:
            while True:
                k = cv2.waitKey(1)

                if k == 32:
                    break

                elif k == 97:
                    self._add_annotation("lights", seq, "green")

                elif k == 50:
                    self._add_annotation("lights", seq, "orange")

                elif k == 51:
                    self._add_annotation("lights", seq, "red")

                elif k == 52:
                    self._add_annotation("lights", seq, "red_orange")

        elif k == 113:
            raise EOFError

        elif k == 49:
            self._add_annotation("lights", seq, "green")

        elif k == 50:
            self._add_annotation("lights", seq, "orange")

        elif k == 51:
            self._add_annotation("lights", seq, "red")

        elif k == 52:
            self._add_annotation("lights", seq, "red_orange")

    def _before(self):
        self.annotations["lights"] = {
                                "green": [],
                                "orange": [],
                                "red": [],
                                "red_orange": []
                            }


annotator = LightAnnotator()
annotator.run()
