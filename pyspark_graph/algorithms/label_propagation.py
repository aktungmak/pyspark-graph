from pyspark.sql import Column
from pyspark.sql.functions import col, mode

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.algorithms.pregel import Pregel
from pyspark_graph.graph import Graph, ID

LABEL = "label"

class LabelPropagation(Algorithm):
    """
    Starting with an initial label (by default the vertex ID if a label column is not
    provided) each vertex updates its label based on the modal value of its neighbours.
    This is repeated for a fixed number of iterations but may converge earlier.

    :param label_column: A Column expression defining the initial label. If not specified the
                         vertex ID is used instead.
    :param max_iterations: By default 10, may be less if the graph stage converges earler.
    """
    result_schema = StructType([StructField(ID, LongType(), False),
                                StructField(LABEL, LongType(), False)])

    def __init__(self, label_column: Column = None, max_iterations=10):
        self.label_column = label_column
        self.max_iterations = max_iterations

    def run(self, g: Graph):
        p = Pregel(initial_state=col(ID) if self.label_column is None else self.label_column,
                   agg_expr=mode(Pregel.MSG),
                   msg_to_src=None if g.directed else col(Pregel.STATE),
                   msg_to_dst=col(Pregel.STATE),
                   max_iterations=self.max_iterations)
        result = p.run(g)
        return result.select(col(ID), col(Pregel.STATE).alias(LABEL))
