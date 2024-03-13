from unittest import TestCase

import pyspark
from pyspark.sql.functions import col, max as _max, greatest

from pyspark_graph.algorithms import Pregel, ConnectedComponents, WLKernel, LabelPropagation, KatzIndex, TriangleCount
from pyspark_graph.graph import ID
from tests import samples


class SparkTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = pyspark.sql.SparkSession.getActiveSession()
        if cls.spark is None:
            cls.spark = pyspark.sql.SparkSession(pyspark.SparkContext())

    @classmethod
    def tearDownClass(cls):
        cls.spark.sparkContext.stop()


class TestPregel(SparkTest):

    def test_max_value(self):
        g = samples.sample1(self.spark)
        s = Pregel.Pregel(initial_state=col(ID),
                          agg_expr=_max(Pregel.MSG),
                          msg_to_src=col(Pregel.STATE),
                          msg_to_dst=col(Pregel.STATE),
                          update_expr=greatest(Pregel.MSG, Pregel.STATE))
        result = s.run(g).select(ID, Pregel.STATE).collect()
        # TODO check output


class TestConnectedComponents(SparkTest):

    def test_two_components(self):
        g = samples.two_components(self.spark)
        c = ConnectedComponents.ConnectedComponents().run(g)
        r = c.groupBy(ConnectedComponents.COMPONENT).count().collect()
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0][1], 3)
        self.assertEqual(r[1][1], 3)


class TestWLKernel(SparkTest):

    def test_labelled(self):
        g = samples.labelled(self.spark)
        r = WLKernel.WLKernel(label_column="label").run(g)
        self.assertEqual('874ee76d142f4da4b531b6c5543b292278027430', r)

    def test_unlabelled(self):
        g = samples.labelled(self.spark)
        r = WLKernel.WLKernel().run(g)
        self.assertEqual('766dd9dcdfc8de175410960d32ad84e8a6116c4e', r)


class TestLabelPropagation(SparkTest):
    def test_labelled(self):
        g = samples.labelled(self.spark)
        r = LabelPropagation.LabelPropagation(label_column="label", max_iterations=3).run(g)
        r.show()

    def test_unlabelled(self):
        g = samples.labelled(self.spark)
        r = LabelPropagation.LabelPropagation(max_iterations=3).run(g)
        r.show()


class TestKatzIndex(SparkTest):
    def test_run(self):
        g = samples.sample1(self.spark)
        k = KatzIndex.KatzIndex(max_iterations=2)
        k.run(g).show()

        # {0: 0.43046326669401147,
        #  1: 0.5087293033711362,
        #  4: 0.43046326669401147,
        #  2: 0.43046326669401147,
        #  3: 0.43046326669401147}


class TestBSSSConnectedComponents(SparkTest):

    def test_two_components(self):
        self.spark.sparkContext.setCheckpointDir("/tmp/bssscc_test")
        g = samples.two_components(self.spark)
        c = ConnectedComponents.BSSSConnectedComponents().run(g)
        r = c.groupBy(ConnectedComponents.COMPONENT).count().collect()
        self.assertEqual(len(r), 2)
        self.assertEqual(r[0][1], 3)
        self.assertEqual(r[1][1], 3)


class TestTriangleCount(SparkTest):
    def test_two_triangles(self):
        g = samples.sample1(self.spark)
        c = TriangleCount.TriangleCount().run(g)
        self.assertEqual(c, 2)

    def test_two_components(self):
        g = samples.two_components(self.spark)
        c = TriangleCount.TriangleCount().run(g)
        self.assertEqual(c, 1)
