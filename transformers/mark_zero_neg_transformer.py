from pyspark.sql import SparkSession
from pyspark.ml import Transformer
from .aggregated_transformer import get_integer_columns, cast_df_col_to_int, save_aggregated_data
import warnings
import traceback
warnings.filterwarnings('ignore')
spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()


class MarkZeroNegTransformer(Transformer):

    def __init__(self):
        self.aggregated_file_path = 'aggregated/mark_neg_zero.csv'
        pass

    def _transform(self, df):
        """
        Create new flag column that will show either each store contain any zero or negative sales or not
        If a contain zero/negative sales flag will be 1 to related store else flag will be zero
        :param df:
        :return df:
        """
        try:
            agg_pd = df.toPandas()
            temp_cols = get_integer_columns(df)
            temp = agg_pd[temp_cols]
            temp['flag'] = (temp == 0).sum(axis=1)
            temp[temp.flag < 1] = 1
            temp[temp.flag > 1] = 0
            agg_pd['flag'] = temp.flag
            agg_df = spark.createDataFrame(agg_pd)
            agg_df = cast_df_col_to_int(agg_df)
            # save_aggregated_data(agg_df, self.aggregated_file_path)
            return agg_df
        except:
            traceback.print_exc()
            return False

