from pyspark.sql import DataFrame

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph, ID


class ShortestPaths(Algorithm):
    """
    Calculate the shortest paths from all vertices to a selection of
    landmarks, up to a maximum length.
    """

    def __init__(self, landmarks: DataFrame, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.landmarks = landmarks
        if ID not in self.landmarks.columns:
            raise ValueError(f"no column '{ID}' in landmarks dataframe")

    def run(self, g: Graph) -> DataFrame:
        p = Pregel(initial_state="{ID: []} if landmark else {}",
                   agg_expr="for each key in the map, take the shortest",
                   msg_to_src=None if g.directed else col(Pregel.STATE),
                   msg_to_dst=col(Pregel.STATE),
                   max_iterations=self.max_iterations)
        result = p.run(g)
        return result.select(col(ID), col(Pregel.STATE).alias(LABEL))
