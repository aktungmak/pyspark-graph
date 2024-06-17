from typing import Optional

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph, ID

MSG = "message"
SRC_VERTEX_PREFIX = "src_vertex_"
DST_VERTEX_PREFIX = "dst_vertex_"


class AggregateMessages(Algorithm):
    """
    Iteratively aggregate messages from vertex to vertex.
    Maybe remove given poor iterative performance in Spark?
    """

    result_schema = StructType([StructField(ID, LongType, False), StructField(MSG, Any, False)])

    def __init__(self, agg: Column, to_src: Optional[Column] = None, to_dst: Optional[Column] = None):
        if to_src is None and to_dst is None:
            raise ValueError("need at least one of to_src or to_dst")
        self.agg = agg
        self.to_src = to_src
        self.to_dst = to_dst

    def src_col(self, col_name: str) -> Column:
        return col(self.SRC_VERTEX_PREFIX + col_name)

    def dst_col(self, col_name: str) -> Column:
        return col(self.DST_VERTEX_PREFIX + col_name)

    def run(self, g: Graph) -> DataFrame:
        triplets = g.triplets(self.SRC_VERTEX_PREFIX, self.DST_VERTEX_PREFIX)
        if self.to_src and self.to_dst:
            to_src = triplets.select(self.to_src.alias(self.MSG), col(self.SRC_VERTEX_PREFIX + ID).alias(ID))
            to_dst = triplets.select(self.to_dst.alias(self.MSG), col(self.DST_VERTEX_PREFIX + ID).alias(ID))
            result = to_src.union(to_dst)
        elif self.to_src:
            result = triplets.select(self.to_src.alias(self.MSG), col(self.SRC_VERTEX_PREFIX + ID).alias(ID))
        elif self.to_dst:
            result = triplets.select(self.to_dst.alias(self.MSG), col(self.SRC_VERTEX_PREFIX + ID).alias(ID))
        return result.groupBy(ID).agg(self.agg)
