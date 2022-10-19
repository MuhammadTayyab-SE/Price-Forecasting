import unittest
from pyspark.sql.session import SparkSession
import math
from feature_transformers.log_transformer import LogTransformer
from feature_transformers.lag_feature import LagTransformer
import pyspark.sql.functions as f


class MyTestCase(unittest.TestCase):

    def test_log_transformer(self):
        # create custom dataframe for testing
        columns = ["sales"]
        data = [20000, 100000, 3000, 1]
        spark = SparkSession.builder.appName('Unit Test').getOrCreate()
        rdd = spark.sparkContext.parallelize(data)
        df = rdd.map(lambda x: (x,)).toDF(columns)

        # applying log transformation using LogTransformer class on custom dataframe
        transformer = LogTransformer(column=['sales'])
        df = transformer.transform(df)

        # converting result transformed result into list
        cal_data = df.select('sales').rdd.flatMap(lambda x: x).collect()

        # comparing the results
        self.assertEqual(cal_data, [math.log10(x) for x in data])

    def test_lag_transformer(self):
        # reading dataframe from csv file
        spark = SparkSession.builder.master("local[5]").appName('MLE Test Case').getOrCreate()
        df = spark.read.csv('../test_dataset/sales_data.csv', inferSchema=True, header=True)

        column_count = len(df.columns)

        # applying lag transformation on dataset using custom transformer
        offset = 2
        lag_transformer = LagTransformer(offset=offset, lagged_column='sales')
        df = lag_transformer.transform(df)

        # calculating nans in sales column
        trans = df.select(f.count(f.when(f.isnan('sales') | f.col('sales').isNull(), 'sales')).alias('sales'))

        # converting sales column data into list
        count = trans.select('sales').rdd.flatMap(lambda x: x).collect()[0]
        # if lag transformation works correctly there must be multiple nans in dataset

        with self.subTest():
            self.assertGreater(count, 0)

        #  check either new lagged column added or not
        with self.subTest():
            self.assertEqual(len(df.columns), column_count + offset)


if __name__ == '__main__':
    unittest.main()
