from pyspark.sql.session import SparkSession
from pyspark.ml import Transformer
from .aggregated_transformer import get_schema
import math
import warnings

warnings.filterwarnings('ignore')


# UDF function for splitting data into train and test
def seprate_train_test(df):
    """
        New column will be added named split that contain value of train or test
    :param df:
    :return df:
    """
    count = df['store_id'].count()
    df['split'] = 'train'
    df['index'] = range(0, count)

    count = df['store_id'].count()
    train_ratio = math.floor(count * 0.7)
    df.loc[df['index'] > train_ratio, 'split'] = 'test'
    df.drop('index', axis=1, inplace=True)
    return df


class TrainTestTransformer(Transformer):

    def __init__(self):
        self.spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()
        self.schema = str()

    def _transform(self, df):

        # get schema of the returned table form UDF
        self.schema = get_schema(df)
        self.schema += ', split string'

        # group data of store, dept and apply UDF
        split_df = df.groupBy(['store_id', 'dept_id']).applyInPandas(seprate_train_test, schema=self.schema)
        cols = split_df.columns
        cols.remove('split')

        # separate train test rows
        train = split_df.where(split_df.split == "train").select(cols)
        test = split_df.filter(split_df.split == "test").select(cols)

        return train, test

