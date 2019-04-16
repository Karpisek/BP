class ObjectSize:
    def __init__(self, w, h, info=None):
        self.width = w
        self.height = h

        if info is not None:
            self.width *= info.width
            self.height *= info.height

            self.square_size = (self.width * self.height) / (info.width * info.height)

    def resized(self, coef):
        return ObjectSize(self.width * coef, self.height * coef)
