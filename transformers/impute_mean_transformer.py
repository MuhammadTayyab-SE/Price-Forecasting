import pyspark.sql.functions as F
from pyspark.ml import Transformer
from .aggregated_transformer import get_integer_columns, cast_df_col_to_int, save_aggregated_data
import warnings
warnings.filterwarnings('ignore')


class ImputeMeanTransformer(Transformer):

    def __init__(self):
        self.mean_file_path = 'aggregated/imputed_mean.csv'
        pass

    def _transform(self, df):
        df = cast_df_col_to_int(df)
        col_names = get_integer_columns(df)

        if 'flag' in df.columns:
            agg_store_df = df.drop('flag').groupby('store_id').mean()
            col_names = col_names[:len(col_names) - 1]
        else:
            col_names = col_names[:len(col_names)]
            agg_store_df = df.groupby('store_id').mean()

        agg_store_cols = ['store_id']
        agg_store_cols.extend(col_names)

        agg_store_df = agg_store_df.toDF(*agg_store_cols)
        # save_aggregated_data(df=agg_store_df, path=self.mean_file_path)

        return agg_store_df
