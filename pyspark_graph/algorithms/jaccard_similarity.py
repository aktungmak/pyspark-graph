from pyspark.sql import DataFrame
from pyspark.sql.functions import col, size, array_intersect, array_union

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph, ID, ADJ


class JaccardSimilarity(Algorithm):
    name = "jaccard_similarity"

    def run(self, g: Graph) -> DataFrame:
        return g.adjacency.alias("a") \
            .join(g.adjacency.alias("b"),
                  col(f"a.{ID}") != col(f"b.{ID}")) \
            .select(f"a.{ID}", f"b.{ID}",
                    (size(array_intersect(f"a.{ADJ}", f"b.{ADJ}")) /
                     size(array_union(f"a.{ADJ}", f"b.{ADJ}")))
                    .alias(self.name))
