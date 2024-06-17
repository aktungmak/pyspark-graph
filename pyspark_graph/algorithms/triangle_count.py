from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph
from pyspark_graph.util import match_structure, order_edges


class TriangleCount(Algorithm):
    def run(self, g: Graph) -> int:
        return match_structure(order_edges(g.edges),
                               [("a", "b"), ("b", "c"), ("a", "c")]).count()
