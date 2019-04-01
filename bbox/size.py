class ObjectSize:
    def __init__(self, w, h, info=None):
        self.width = w
        self.height = h

        if info is not None:
            self.width *= info.width
            self.height *= info.height
