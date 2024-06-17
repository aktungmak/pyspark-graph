from pyspark.sql import DataFrame
from pyspark.sql.functions import col, size, array_intersect, least
from pyspark.sql.types import LongType, StructField, StructType

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph, ID, ADJ


class OverlapCoefficient(Algorithm):
    VERTEXA = "vertex_a"
    VERTEXB = "vertex_b"
    OVERLAP_COEF = "overlap_coefficient"
    result_schema = StructType([StructField(VERTEXA, LongType(), False),
                                StructField(VERTEXB, LongType(), False),
                                StructField(OVERLAP_COEF, LongType(), False)])

    def run(self, g: Graph) -> DataFrame:
        return g.adjacency.alias("a") \
            .join(g.adjacency.alias("b"),
                  col(f"a.{ID}") != col(f"b.{ID}")) \
            .select(col(f"a.{ID}").alias(self.VERTEXA),
                    col(f"b.{ID}").alias(self.VERTEXB),
                    (size(array_intersect(f"a.{ADJ}", f"b.{ADJ}")) /
                     least(size(f"a.{ADJ}"), size(f"b.{ADJ}")))
                    .alias(self.OVERLAP_COEF))
