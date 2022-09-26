import unittest
from pyspark.sql.session import SparkSession
import math
from feature_transformers.log_transformer import LogTransformer
from feature_transformers.lag_feature import LagTransformer
import pyspark.sql.functions as F


class MyTestCase(unittest.TestCase):

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


    def test_lag_trasformer(self):
        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()
        df = spark.read.csv('test_dataset/sales_data.csv', inferSchema=True, header=True)

        lag_transformer = LagTransformer(2)
        df = lag_transformer.transform(df)
        trans = df.select(F.count(F.when(F.isnan('sales') | F.col('sales').isNull(), 'sales')).alias('sales'))
        count = trans.select('sales').rdd.flatMap(lambda x: x).collect()[0]
        self.assertGreater(count, 0)


if __name__ == '__main__':
    unittest.main()