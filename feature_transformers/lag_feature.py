from pyspark.sql.window import Window
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Transformer
import warnings
import functools

warnings.filterwarnings('ignore')

class LagTransformer(Transformer):

    def __init__(self, offset):
        self.offset = offset
        self.spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()


    def unionAll(self, dfs):
        if len(dfs[0].columns) != 0:
            return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
        else:
            return dfs[1]

    def _transform(self, df):

        stores = df.select('store_id').distinct().orderBy('store_id').collect()
        depts = df.select('dept_id').distinct().orderBy('dept_id').collect()
        data = [()]
        final_df = self.spark.createDataFrame(data)

        for store in stores:
            for dept in depts:
                grouped = df.where((df.store_id == store['store_id']) & (df['dept_id'] == dept['dept_id']))
                windowSpec = Window.partitionBy(['store_id']).orderBy(['dept_id'])
                final_df = self.unionAll([final_df, grouped.withColumn("lag_sales", F.lag("sales", self.offset).over(windowSpec))])
        return final_df
