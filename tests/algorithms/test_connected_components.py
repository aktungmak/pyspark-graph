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


class TestBSSSConnectedComponents(SparkTest):

    def test_two_components(self):
        self.spark.sparkContext.setCheckpointDir("/tmp/bssscc_test")
        g = samples.two_components(self.spark)
        c = ConnectedComponents.BSSSConnectedComponents().run(g)
        r = c.groupBy(ConnectedComponents.COMPONENT).count().collect()
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0][1], 3)
        self.assertEqual(r[1][1], 3)
