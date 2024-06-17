from pyspark.sql import DataFrame
from pyspark.sql.functions import max as _max

from pyspark_graph import matrix
from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph, SRC, DST
from pyspark_graph import graph_to_adjacency_matrix


class KatzIndex(Algorithm):
    """
    TODO
    Calculate the Katz index of each pair of vertices, which represents the number
    of paths up to max_iterations long there are between pairs of vertices.
    """
    INDEX = "katz_index"

    def __init__(self, beta: float = 1.0, tolerance: float = None, max_iterations: int = 10):
        self.beta = beta
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> DataFrame:
        a = graph_to_adjacency_matrix(g)
        for _ in range(self.max_iterations):
            a *= a
            a.df.show()
            if self.tolerance is not None:
                # check if average difference is below tolerance
                max_delta = a.df.agg(_max(matrix.VAL)).first()[0]
                if max_delta < self.tolerance:
                    break

        return a.df.withColumnsRenamed({matrix.ROW: SRC,
                                        matrix.COL: DST,
                                        matrix.VAL: self.INDEX})
