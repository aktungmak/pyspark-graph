from pyspark.sql.functions import col, max as _max, greatest

from pyspark_graph.algorithms import pregel
from pyspark_graph.graph import ID
from tests import samples
from tests.spark_test import SparkTest


class TestPregel(SparkTest):

    def test_max_value(self):
        g = samples.sample1(self.spark)
        s = Pregel.Pregel(initial_state=col(ID),
                          agg_expr=_max(Pregel.MSG),
                          msg_to_src=col(Pregel.STATE),
                          msg_to_dst=col(Pregel.STATE),
                          update_expr=greatest(Pregel.MSG, Pregel.STATE))
        result = s.run(g).select(ID, Pregel.STATE).collect()
        # TODO check output
