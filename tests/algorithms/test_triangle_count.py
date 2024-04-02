from tests import samples
from tests.spark_test import SparkTest


class TestTriangleCount(SparkTest):
    def test_two_triangles(self):
        g = samples.sample1(self.spark)
        c = TriangleCount.TriangleCount().run(g)
        self.assertEqual(c, 2)

    def test_two_components(self):
        g = samples.two_components(self.spark)
        c = TriangleCount.TriangleCount().run(g)
        self.assertEqual(c, 1)
