from pyspark.ml import Transformer
from .aggregated_transformer import get_orignal_columns
import traceback
import warnings
warnings.filterwarnings('ignore')


class ImputeMeanTransformer(Transformer):

    def __init__(self):
        pass

    def _transform(self, df):
        try:
            agg_store_df = df.groupby(['store_id', 'month']).avg().orderBy(['store_id', 'month'])
            new_cols = get_orignal_columns(agg_store_df.columns)
            print(new_cols)
            aggregated_df = agg_store_df.toDF(*new_cols)
            return aggregated_df
        except:
            traceback.print_exc()
            return None
