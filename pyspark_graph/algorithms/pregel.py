import typing

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit

from pyspark_graph.algorithms import Algorithm
from pyspark_graph.graph import Graph, DST, SRC, ID
from pyspark_graph.util import ne_null_safe, multiple_union


class Pregel(Algorithm):
    """
    initial_state: the initial state of the vertex before iteration starts, can use all vertex columns
    update_expr: executed at the start of each iteration to update state, can use all vertex columns plus MSG
    agg_expr: an expression on Pregel.MSG to aggregate all messages arriving at a vertex
    changed: a function that takes the old and new state and returns True if the state changed
    msg_to_dst: will be sent to all out-neighbours after state update, can use all vertex columns
    msg_to_dst: will be sent to all in-neighbours after state update, can use all vertex columns
    max_iterations: limit number of supersteps if no convergence
    """
    STATE = "state"
    OLD_STATE = "old_state"
    MSG = "message"  # column containing aggregated messages

    def __init__(self,
                 initial_state: Column,
                 agg_expr: Column,
                 msg_to_src: Column = None,
                 msg_to_dst: Column = None,
                 update_expr: Column = None,
                 comparison: typing.Callable = ne_null_safe,
                 max_iterations: int = 10):
        if msg_to_src is None and msg_to_dst is None:
            raise ValueError("need at least one of msg_to_src or msg_to_dst")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        self.initial_state = initial_state
        self.agg_expr = agg_expr
        self.msg_to_src = msg_to_src
        self.msg_to_dst = msg_to_dst
        self.update_expr = col(self.MSG) if update_expr is None else update_expr
        self.comparison = comparison
        self.max_iterations = max_iterations

    def run(self, g: Graph) -> DataFrame:
        state = g.vertices.withColumns({self.STATE: self.initial_state,
                                        self.OLD_STATE: lit(None)})
        changed = state
        for i in range(self.max_iterations):
            # send msgs
            message_dfs = []
            if self.msg_to_src is not None:
                message_dfs.append(self._send(changed, g.edges, self.msg_to_src, DST, SRC))
            if self.msg_to_dst is not None:
                message_dfs.append(self._send(changed, g.edges, self.msg_to_dst, SRC, DST))
            messages = multiple_union(message_dfs)

            # aggregate messages
            agg_messages = messages.groupBy(ID).agg(self.agg_expr.alias(self.MSG))

            # update vertex state
            updated = agg_messages.join(state, ID) \
                .withColumns({self.OLD_STATE: col(self.STATE),
                              self.STATE: self.update_expr}) \
                .drop(self.MSG)
            # DataFrame does not support insert so we use antijoin+union
            not_updated = state.join(messages, ID, "anti")
            state = updated.union(not_updated)

            # check for termination
            changed = updated.filter(self.comparison(col(self.STATE), col(self.OLD_STATE)))
            if changed.isEmpty():
                break

        return state

    def _send(self,
              changed_vertices: DataFrame,
              edges: DataFrame,
              msg_expr: Column,
              _from: str,
              to: str) -> DataFrame:
        """Take a dataframe of changed vertices, apply the send expression
         and then join the result through the edge to the destination.
         Result is a DataFrame of recipients and messages."""
        return changed_vertices \
            .select(col(ID).alias(_from),
                    msg_expr.alias(self.MSG)) \
            .join(edges, _from) \
            .select(col(to).alias(ID), col(self.MSG))
