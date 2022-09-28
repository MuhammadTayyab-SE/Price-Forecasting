from pyspark.ml import Transformer
from transformers.aggregated_transformer import get_schema
import traceback
import warnings
warnings.filterwarnings('ignore')


# fill nans with the mean of respective group

def fill_na(df):
    """ This function full nan values in sales with the mean of that group
    :param: df
    :return: df that contain no nans
    """
    df['sales'] = df['sales'].fillna(df['sales'].mean())
    df = df.round(2)
    return df


class ImputeMeanTransformer(Transformer):

    def __init__(self):
        super().__init__()
        self.schema = str()

    def _transform(self, df):
        try:
            self.schema = get_schema(df)

            # replace nan sales with mean of that group using UDF
            agg_df = df.groupby(['store_id', 'dept_id']).applyInPandas(fill_na, schema=self.schema)
            return agg_df
        except (Exception,):
            traceback.print_exc()
            return None
