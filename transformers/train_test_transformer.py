import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from pyspark.ml import Transformer
import math
import warnings
import functools


warnings.filterwarnings('ignore')


class TrainTestTransformer(Transformer):

    def __init__(self):
        self.spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()

    def unionAll(self, dfs):
        if len(dfs[0].columns) != 0:
            return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
        else:
            return dfs[1]

    def _transform(self, df):
        data = [()]
        final_train_df = self.spark.createDataFrame(data)
        final_test_df = self.spark.createDataFrame(data)

        stores = df.select('store_id').distinct().orderBy('store_id').collect()
        depts = df.select('dept_id').distinct().orderBy('dept_id').collect()
        dataset_count = df.where((df.store_id == 'CA_1') & (df.dept_id == 'FOODS_1')).count()
        train_ratio = math.floor(dataset_count * 0.7)

        for store in stores:
            for dept in depts:
                dataset = df.where((df.store_id == store['store_id']) & (df.dept_id == dept['dept_id']))

                # adding new column named index for split
                dataset = dataset.withColumn("index", F.monotonically_increasing_id())

                # computing train test ratio

                # train test split
                training_df = dataset.filter(dataset.index < train_ratio)
                testing_df = dataset.filter(dataset.index >= train_ratio)

                training_df = training_df.drop(F.col('index'))
                testing_df = testing_df.drop(F.col('index'))

                final_train_df = self.unionAll([final_train_df, training_df])
                final_test_df = self.unionAll([final_test_df, testing_df])

        return final_train_df, final_test_df

