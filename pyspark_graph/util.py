from functools import reduce

from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import col, least, greatest

from .graph import SRC, DST


def multiple_join(dfs: list[DataFrame]) -> DataFrame:
    def join(left, right):
        on = list(set(left.columns) & set(right.columns))
        return left.join(right, on=on)

    return reduce(join, dfs)


def multiple_union(dfs: list[DataFrame]) -> DataFrame:
    def union(left, right):
        return left.union(right)

    return reduce(union, dfs)


def ne_null_safe(x: Column, y: Column) -> Column:
    return ~x.eqNullSafe(y)


def match_structure(edges: DataFrame, match: list[tuple[str, str]]):
    if len(match) == 0:
        raise ValueError("match list must not be empty")
    pos_dfs = [edges.select(col(SRC).alias(s), col(DST).alias(d)) for s, d in match]
    return multiple_join(pos_dfs)


def order_edges(edges: DataFrame) -> DataFrame:
    """Reorder edges to be ascending, remove self loops and drop duplicates"""
    return edges.filter(f"{SRC} != {DST}") \
        .select(least(col(SRC), col(DST)).alias(SRC), greatest(col(SRC), col(DST)).alias(DST)) \
        .dropDuplicates()
