class Repository:
    def ready(self):
        raise NotImplementedError

    def select_manually(self, image):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError
