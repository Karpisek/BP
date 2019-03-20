class ObjectSize:
    def __init__(self, w, h):
        self.width = w
        self.height = h

        self.is_relative = True

    def convert_to_fixed(self, info):
        if self.is_relative:
            self.width *= info.width
            self.height *= info.height

            self.is_relative = False
