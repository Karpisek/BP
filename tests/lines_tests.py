import unittest
from pc_lines.line import Line, SamePointError, NoIntersectionError, NotOnLineError


class LinearFunctionTests(unittest.TestCase):
    def test_horizontal(self):
        self.assertTrue(Line((0, 1), (2, 1)).horizontal)
        self.assertFalse(Line((0, 1), (2, 1)).vertical)

    def test_vertical(self):
        self.assertFalse(Line((0, 1), (2, 1)).vertical)
        self.assertTrue(Line((0, 1), (0, 2)).vertical)

    def test_same_points(self):
        with self.assertRaises(SamePointError):
            Line((3, 3), (3, 3))

    def test_angle_90(self):
        l1 = Line((0, 1), (0, 0))
        l2 = Line((0, 0), (1, 0))

        self.assertEqual(l1.angle(l2), 90.)

    def test_angle_019(self):
        l1 = Line((1, 2), (3, 5))
        l2 = Line((3, 4), (2, 5))

        self.assertAlmostEqual(l1.angle(l2), 78.7, 1)

    def test_point(self):
        l1 = Line((1, 1), (2, 2))

        self.assertTrue(l1.on_line((3, 3)))
        self.assertFalse(l1.on_line((-3, 3)))

    def test_intersection(self):
        l1 = Line((0, 0), (1, 0))
        l2 = Line((0, 0), (0, 1))

        self.assertEqual(l1.intersection(l2), (0, 0))

    def test_parallel(self):
        l1 = Line((-3, 2), (-3, 3))
        l2 = Line((-2, 3), (-2, 4))
        with self.assertRaises(NoIntersectionError):
            l1.intersection(l2)

    def test_same(self):
        l1 = Line((-3, 2), (-3, 3))
        l2 = Line((-2, 2), (-2, 3))
        with self.assertRaises(NoIntersectionError):
            l1.intersection(l2)

    def test_angle(self):
        l1 = Line((0, 0), (0, 1))
        l2 = Line((0, 0), (1, 0))
        l3 = Line((0, 0), (1, 1))
        self.assertEqual(l1.angle(l2), 90)
        self.assertAlmostEqual(l1.angle(l3), 45, 4)

        l2.general_equation()

    def test_point_distance(self):
        l1 = Line((0, 0), (0, 1))
        point1 = (0, 0)
        point2 = (1, 0)

        self.assertEqual(l1.point_distance(point1), 0)
        self.assertEqual(l1.point_distance(point2), 1)

    #def test_normal(self):
    #     l1 = LinearFunction((0, 0), (0, 1))
    #     l2 = LinearFunction((0, 0), (1, 0))
    #     p1 = (1, 0)
    #     self.assertEqual(l2.distance(p1), 0)
    #     self.assertEqual(l1.distance(p1), 1)
    #
    def test_parse_to_line(self):
        l1 = Line((0, 0), (1, 0))
        self.assertEqual(l1.find_coordinate(x=2), (2, 0))

        l1 = Line((0, 1), (0, 0))
        with self.assertRaises(NotOnLineError):
            l1.find_coordinate(x=2)

        l1 = Line((0, 0), (5, 0))
        with self.assertRaises(NotOnLineError):
            l1.find_coordinate(y=2)
