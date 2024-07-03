from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, min as _min, least, explode, array, struct, greatest, count, sum as _sum, when, \
    array_agg
from pyspark.sql.types import DecimalType

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.algorithms.pregel import Pregel
from pyspark_graph.graph import Graph, ID, SRC, DST

COMPONENT = "component"
ALGO_PREGEL = "pregel"
ALGO_ALTERNATING = "alternating"
ALGO_LAPLACIAN = "laplacian"


# TODO implement laplacian algorithm

class ConnectedComponents(Algorithm):
    """
    Identify connected components if the graph is undirected and strongly connected components
    if the graph is directed.

    :param max_iterations:
    """

    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations

    def run(self, g) -> DataFrame:
        p = Pregel(initial_state=col(ID),
                   agg_expr=_min(Pregel.MSG),
                   msg_to_src=col(Pregel.STATE) if not g.directed else None,
                   msg_to_dst=col(Pregel.STATE),
                   update_expr=least(Pregel.MSG, Pregel.STATE),
                   max_iterations=self.max_iterations)
        return p.run(g).select(col(ID), col(Pregel.STATE).alias(COMPONENT))


class AlternatingConnectedComponents(Algorithm):
    MIN_NBR = "min_nbr"

    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations

    def symmetric_edges(self, edges: DataFrame) -> DataFrame:
        """add extra edges to ensure SRC->DST and DST->SRC are present"""
        return edges.union(edges.select(col(DST).alias(SRC),
                                        col(SRC).alias(DST)))

    def minimum_neighbour(self, edges: DataFrame) -> DataFrame:
        """find the minimum neighbour for each SRC vertex"""
        return edges.withColumn(self.MIN_NBR,
                                least(col(SRC), _min(DST).over(Window.partitionBy(SRC))).alias(DST))

    def large_star(self, edges: DataFrame) -> DataFrame:
        """connect minimum neighbour to all neighbours > SRC, not including SRC"""
        edges = self.symmetric_edges(edges)
        edges = self.minimum_neighbour(edges)
        return (edges.where(col(DST) > col(SRC)).select(col(DST).alias(SRC),
                                                        col(self.MIN_NBR).alias(DST)))

    def orient_edges(self, edges: DataFrame) -> DataFrame:
        """ensure SRC > DST for all edges"""
        return (edges.select(greatest(col(SRC), col(DST)).alias(SRC),
                             least(col(SRC), col(DST)).alias(DST)))

    def small_star(self, edges: DataFrame) -> DataFrame:
        """connect minimum neighbour to all neighbours <= SRC, including SRC"""
        edges = self.orient_edges(edges)
        edges = self.minimum_neighbour(edges)
        return (edges.select(col(DST).alias(SRC),
                             col(self.MIN_NBR).alias(DST))
                .union(edges.select(col(SRC),
                                    col(self.MIN_NBR).alias(DST))))

    def run(self, graph: Graph) -> DataFrame:
        prev_sum = 0
        edges = graph.edges.select(SRC, DST)
        for i in range(self.max_iterations):
            edges = self.large_star(edges)
            edges = self.small_star(edges)
            edges = edges.distinct()

            min_nbr_sum = edges.groupBy().sum(DST).collect()[0][0]
            print(f"iteration {i} complete, sum {min_nbr_sum}")
            if prev_sum == min_nbr_sum:
                break
            else:
                prev_sum = min_nbr_sum
        else:
            print(f"Connected Components did not terminate after {self.max_iterations} iterations!")
        return edges.select(col(SRC).alias(ID), col(DST).alias(COMPONENT))
