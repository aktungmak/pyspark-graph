from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, array, array_contains, array_append
from pyspark.sql.types import StructType, StructField, LongType, ArrayType

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph, SRC, DST, ID, EDGE_ID

START = "start"
END = "end"
EDGES = "edges"
VERTICES = "vertices"


class BreadthFirstSearch(Algorithm):
    """
    Perform breadth-first search from a specified set of source vertices to a destination set of vertices,
    while optionally limiting the edges that can be traversed.
    The search is performed iteratively and is by default limited to 10 iterations.
    Returns start, end and arrays of the edges and vertices traversed.
    """
    result_schema = StructType([StructField(START, LongType(), False), StructField(END, LongType(), False),
                                StructField(EDGES, ArrayType(LongType(), False), False),
                                StructField(VERTICES, ArrayType(LongType(), False), False)])

    def __init__(self, start_expr: Column, end_expr: Column, edge_expr: Column = "true", max_iterations: int = 10):
        self.start_expr = start_expr
        self.end_expr = end_expr
        self.edge_expr = edge_expr
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> DataFrame:
        if g.directed:
            edges = g.edges
        else:
            reverse = g.edges.withColumns({SRC: DST, DST: SRC})
            edges = g.edges.union(reverse).distinct()

        start = g.vertices.filter(self.start_expr)
        end = g.vertices.filter(self.end_expr)
        edges = edges.filter(self.edge_expr)

        # check for trivial empty result
        if start.isEmpty() or edges.isEmpty() or end.isEmpty():
            return g.spark.createDataFrame([], self.result_schema)

        horizon = "horizon"
        paths = start.select(col(ID).alias(self.START),
                             col(ID).alias(horizon),
                             array().alias(self.EDGES),
                             array(col(ID)).alias(self.VERTICES))
        for _ in range(self.max_iterations):
            # check if we reached end or ran out of paths
            result = paths.join(end, paths[horizon] == end[ID])
            if result.head(1) or paths.isEmpty():
                break
            # otherwise extend horizon
            paths = paths.join(edges, (edges[SRC] == paths[horizon]) \
                               & ~array_contains(paths[self.EDGES], edges[EDGE_ID])) \
                .select(col(self.START),
                        col(DST).alias(horizon),
                        array_append(col(self.EDGES), col(EDGE_ID)).alias(self.EDGES),
                        array_append(col(self.VERTICES), col(DST)).alias(self.VERTICES))
        else:
            print("max_iterations reached")
            return g.spark.createDataFrame([], self.result_schema)
        return result.select(col(self.START),
                             col(ID).alias(self.END),
                             col(self.EDGES),
                             col(self.VERTICES))
