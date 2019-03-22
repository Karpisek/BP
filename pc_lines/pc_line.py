from .line import Line, NoIntersectionError
import params


class PcLines:
    def __init__(self, width):
        self.delta = width
        self.t_space = []
        self.s_space = []

    def ransac(self, points):
        raise NotImplementedError

    def pc_line_from_points(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        magnitude = Line(point1, point2).magnitude

        if magnitude < params.CALIBRATOR_FLOW_THRESHOLD:
            return

        l1_s = Line((0, x1), (self.delta, y1))
        l1_t = Line((-self.delta, -y1), (0, x1))

        l2_s = Line((0, x2), (self.delta, y2))
        l2_t = Line((-self.delta, -y2), (0, x2))

        u = None
        v = None

        try:
            u, v = l1_s.intersection(l2_s)
        except NoIntersectionError:
            pass

        try:
            u, v = l1_t.intersection(l2_t)
        except NoIntersectionError:
            pass

        if u is not None and v is not None:
            if u < 0:
                self.t_space.append(((u, v), magnitude))
            else:
                self.s_space.append(((u, v), magnitude))





