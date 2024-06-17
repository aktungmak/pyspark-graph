from unittest import TestCase

import pyspark

from tests.samples import sample1


class TestMultipleJoin(TestCase):
    def test_successful(self):
        self.fail()


class TestMatchStructure(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = pyspark.sql.SparkSession.getActiveSession()
        if cls.spark is None:
            cls.spark = pyspark.sql.SparkSession(pyspark.SparkContext())
        cls.g = sample1(cls.spark)

    @classmethod
    def tearDownClass(cls):
        cls.spark.sparkContext.stop()

    def test_no_edges(self):
        self.assertRaises(ValueError, pyspark_graph.util.match_structure(self.g.edges, []))

    def test_one_edge(self):
        self.assertEqual(22, pyspark_graph.util.match_structure(self.g.edges, ["a", "b"]))

    def test_no_matching_edges(self):
        self.fail()

    def test_successful(self):
        self.fail()
