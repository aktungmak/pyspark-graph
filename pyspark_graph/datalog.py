from typing import Optional, Union

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import lit

from graph import Graph, ID, SRC, DST
from util import multiple_join


class Vertex:
    def __init__(self, name: str, condition: Column | str):
        self.name = name
        self.condition = condition

    def apply(self, g: Graph) -> DataFrame:
        return g.vertices.filter(self.condition).withColumnRenamed(ID, self.name)


class Edge:
    def __init__(self, src: str, dst: str, condition: Optional[Column | str] = None):
        self.src = src
        self.dst = dst
        self.condition = condition if condition is not None else lit(True)

    def apply(self, g: Graph) -> DataFrame:
        return g.edges.filter(self.condition).withColumnsRenamed({SRC: self.src, DST: self.dst})


Rule = Union[Vertex | Edge]


class DatalogQuery:
    def __init__(self, projection: list[Column | str] = [], premises: list[Rule] = [],
                 negated_premises: Optional[list[Rule]] = None):
        self._projection = projection
        self._premises = premises
        self._negated_premises = negated_premises

        # TODO validate range restriction property

    def apply(self, g: Graph):
        pos_dfs = [premise.apply(g) for premise in self._premises]
        pos_df = multiple_join(pos_dfs)
        if self._negated_premises is None:
            return pos_df
        else:
            neg_dfs = [premise.apply(g) for premise in self._negated_premises]
            neg_df = multiple_join(neg_dfs)
            return pos_df.join(neg_df, how="anti").select(self._projection)
