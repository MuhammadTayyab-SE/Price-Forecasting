from pyspark.ml import Transformer
import pandas as pd
from .aggregated_transformer import getSchema
import traceback
import warnings
warnings.filterwarnings('ignore')


# fill nans with the mean of respective group

def fill_na(df):
    df['sales'] = df['sales'].fillna(df['sales'].mean())
    df = df.round(2)
    return df


class ImputeMeanTransformer(Transformer):

    def __init__(self):
        self.schema = str()

    def _transform(self, df):
        try:
            self.schema = getSchema(df)

            # replace nan sales with mean of that group using UDF
            agg_df = df.groupby(['store_id', 'dept_id']).applyInPandas(fill_na, schema=self.schema)
            return agg_df
        except:
            traceback.print_exc()
            return None
