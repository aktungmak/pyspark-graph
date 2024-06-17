from typing import Optional, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import lit

from .graph import Graph, ID, SRC, DST
from .util import multiple_join


class Vertex:
    def __init__(self, name: str, condition: Optional[Column | str] = None, **bindings):
        self.name = name
        self.condition = condition if condition is not None else lit(True)
        self.bindings = bindings

    def apply(self, g: Graph) -> DataFrame:
        return g.vertices \
            .filter(self.condition) \
            .withColumnRenamed(ID, self.name) \
            .withColumnsRenamed(self.bindings)


class Edge:
    def __init__(self, src: str, dst: str, condition: Optional[Column | str] = None):
        self.src = src
        self.dst = dst
        self.condition = condition if condition is not None else lit(True)

    def apply(self, g: Graph) -> DataFrame:
        return g.edges \
            .filter(self.condition) \
            .withColumnsRenamed({SRC: self.src, DST: self.dst})


Rule = Union[Vertex | Edge]


class DatalogQuery:
    """
    Query a graph using a datalog style query structure to do advanced
    motif finding.
    Each Vertex and Edge object defines a subset of the vertices and edges
    in the graph. Additionally, the attributes are labelled with names
    that are used to join together the result.
    By providing a set of negated premises, you can eliminate them from the
    result set.
    """

    def __init__(self,
                 projection: list[Column] = [],
                 premises: list[Rule] = [],
                 negated_premises: Optional[list[Rule]] = None):
        self._projection = projection
        self._premises = premises
        self._negated_premises = negated_premises

        # TODO validate range restriction property

    def apply(self, g: Graph) -> DataFrame:
        pos_dfs = [premise.apply(g) for premise in self._premises]
        result = multiple_join(pos_dfs)
        if self._negated_premises is not None:
            neg_dfs = [premise.apply(g) for premise in self._negated_premises]
            neg_df = multiple_join(neg_dfs)
            result = result.join(neg_df, how="anti")
        return result.select(self._projection)
