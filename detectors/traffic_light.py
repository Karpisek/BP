class TrafficLightsRepository:
    def __init__(self):
        raise NotImplementedError


class TrafficLight:
    def __init__(self, left_top, bottom_right):
        self._left_top = left_top
        self._bottom_right = bottom_right

    def state(self):
        raise NotImplementedError
