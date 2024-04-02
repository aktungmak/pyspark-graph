from unittest import TestCase

import pyspark


class SparkTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = pyspark.sql.SparkSession.getActiveSession()
        if cls.spark is None:
            cls.spark = pyspark.sql.SparkSession(pyspark.SparkContext())

    @classmethod
    def tearDownClass(cls):
        cls.spark.sparkContext.stop()
