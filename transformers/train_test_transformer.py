import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
import pandas as pd
from pyspark.ml import Transformer
import math
import warnings
import functools



warnings.filterwarnings('ignore')


def g(df):
    count = df['store_id'].count()
    df['split'] = 'train'
    df['index'] = range(0, count)

    count = df['store_id'].count()
    train_ratio = math.floor(count * 0.7)
    df.loc[df['index'] >= train_ratio, 'split'] = 'test'
    df.drop('index', axis=1, inplace=True)
    return df


class TrainTestTransformer(Transformer):

    def __init__(self):
        self.spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()
        self.schema =  """
            store_id string,
            dept_id string,
            date string,
            demand double,
            wm_yr_wk double,
            wday double,
            month double,
            year double,
            sell_price double,
            sales double,
            flag int,
            split string
        """

    def unionAll(self, dfs):
        if len(dfs[0].columns) != 0:
            return functools.reduce(lambda df1, df2: df1.union(df2.select(df1.columns)), dfs)
        else:
            return dfs[1]

    def _transform(self, df):
        split_df = df.groupBy(['store_id', 'dept_id']).applyInPandas(g, schema=self.schema)
        cols = split_df.columns
        cols.remove('split')
        train = split_df.where(split_df.split == "train").select(cols)
        test = split_df.filter(split_df.split == "test").select(cols)
        return train, test

