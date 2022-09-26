import unittest
from pyspark.sql.session import SparkSession
import math
from feature_transformers.log_transformer import LogTransformer
from feature_transformers.lag_feature import LagTransformer



class MyTestCase(unittest.TestCase):
    agg_df = None

    def test_log_transformer(self):
        columns = ["sales"]
        data = [20000, 100000, 3000, 1]
        spark = SparkSession.builder.appName('Unit Test').getOrCreate()
        rdd = spark.sparkContext.parallelize(data)
        df = rdd.map(lambda x: (x,)).toDF(columns)

        transformer = LogTransformer()
        df = transformer.transform(df)
        cal_data = df.select('sales').rdd.flatMap(lambda x: x).collect()
        self.assertEqual(cal_data, [math.log10(x) for x in data])






if __name__ == '__main__':
    unittest.main()