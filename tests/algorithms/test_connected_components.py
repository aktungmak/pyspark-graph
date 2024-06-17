from tests import samples
from tests.spark_test import SparkTest


class TestConnectedComponents(SparkTest):

    def test_two_components(self):
        g = samples.two_components(self.spark)
        c = ConnectedComponents.ConnectedComponents().run(g)
        r = c.groupBy(ConnectedComponents.COMPONENT).count().collect()
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0][1], 3)
        self.assertEqual(r[1][1], 3)


class TestAlternatingConnectedComponents(SparkTest):
    def test_symmetrise(self):
        edges = self.spark.createDataFrame([(1, 2), (3, 4), (4, 3)], ["src", "dst"])
        r = ConnectedComponents.AlternatingConnectedComponents.symmetrize(edges).collect()
        self.assertEqual(len(r), 4)

    def test_two_components(self):
        self.spark.sparkContext.setCheckpointDir("/tmp/bssscc_test")
        g = samples.two_components(self.spark)
        c = ConnectedComponents.AlternatingConnectedComponents().run(g)
        r = c.groupBy(ConnectedComponents.COMPONENT).count().collect()
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0][1], 3)
        self.assertEqual(r[1][1], 3)
