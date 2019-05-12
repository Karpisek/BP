class ObjectSize:
    """
    Representing object size. Is able to convert relative sizes into real sizes.
    """

    def __init__(self, w, h, info=None):
        """
        :param w: object width
        :param h: object height
        :param info: InputInfo instance
        """

        self.width = w
        self.height = h

        if info is not None:
            self.width *= info.width
            self.height *= info.height

            self.square_size = (self.width * self.height) / (info.width * info.height)

    def resized(self, coef):
        """
        Returns new resized object size

        :param coef: coef of resize
        :return: new resized object
        """

        return ObjectSize(self.width * coef, self.height * coef)
