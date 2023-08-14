from pyspark.sql import SparkSession

from pyspark_graph.graph import Graph, ID, SRC, DST


def sample1(spark: SparkSession):
    v = spark.createDataFrame([(x,) for x in "abcdef"], [ID])
    e = spark.createDataFrame(
        [("a", "b", 9), ("a", "c", 9), ("b", "d", 9), ("b", "c", 9), ("b", "e", 9), ("e", "d", 9), ("b", "a", 9)],
        [SRC, DST, "nine"])
    return Graph(v, e)


def sample2(spark: SparkSession):
    v = spark.createDataFrame([(x,) for x in "abcdef"], [ID])
    e = spark.createDataFrame(
        [("a", "b", 9), ("b", "c", 9), ("c", "a", 9), ("c", "d", 9), ("d", "e", 9), ("e", "f", 9)],
        [SRC, DST, "nine"])
    return Graph(v, e)
