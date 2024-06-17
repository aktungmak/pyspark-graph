from pyspark.sql.functions import sha1, col, array_join, collect_list

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.algorithms.pregel import Pregel
from pyspark_graph.graph import Graph, ID, DEGREE


class WLKernel(Algorithm):
    """Calculate the Weisfeiler-Lehman kernel for the graph
    Returns a hash that will be the same for two isomorphic graphs.
    By default, the degree of the node will be used as the starting label,
    but any vertex column can be used instead."""

    def __init__(self, hashfunc=sha1, label_column: str = None, max_iterations=3):
        self.hash = hashfunc
        self.label_column = label_column
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> str:
        label = self.label_column
        if label is None:
            vertices = g.vertices.join(g.degrees, ID)
            # TODO find a good way to update the graph
            g = Graph(vertices, g.edges, indexed=True)
            label = DEGREE
        p = Pregel(initial_state=col(label),
                   agg_expr=self.hash(array_join(collect_list(Pregel.MSG), "")),
                   msg_to_src=None if g.directed else col(Pregel.STATE),
                   msg_to_dst=col(Pregel.STATE),
                   max_iterations=self.max_iterations)
        result = p.run(g)
        return result.agg(self.hash(array_join(collect_list(Pregel.STATE), ""))).first()[0]
