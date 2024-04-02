from pyspark import Row

from tests import samples
from tests.spark_test import SparkTest


class TestLabelPropagation(SparkTest):
    def test_labelled(self):
        g = samples.labelled(self.spark)
        r = LabelPropagation.LabelPropagation(label_column="label", max_iterations=3).run(g)
        self.assertEqual([Row(id=0, label=0),
                          Row(id=1, label=1),
                          Row(id=2, label=2),
                          Row(id=5, label=3),
                          Row(id=4, label=3),
                          Row(id=3, label=3)], r.collect())

    def test_unlabelled(self):
        g = samples.labelled(self.spark)
        r = LabelPropagation.LabelPropagation(max_iterations=3).run(g)
        self.assertEqual([Row(id=0, label='a'),
                          Row(id=1, label='b'),
                          Row(id=2, label='c'),
                          Row(id=5, label='d'),
                          Row(id=4, label='d'),
                          Row(id=3, label='d')], r.collect())

    def test_undirected(self):
        # TODO
        pass
