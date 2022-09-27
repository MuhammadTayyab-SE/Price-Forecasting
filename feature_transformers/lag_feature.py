from pyspark.ml import Transformer
import warnings
warnings.filterwarnings('ignore')


# return schema of the dataframe in string format
def getSchema(df):
    types = df.dtypes
    lst = str()
    count = 0
    for col, col_type in types:
        if count != len(types) - 1:
            string = col + " "+col_type + ',' + ' '
        else:
            string = col + " "+col_type + ' '
        lst += string
        count += 1
    return lst


class LagTransformer(Transformer):

    def __init__(self, offset):
        self.offset = offset
        self.schema = str()



    def _transform(self, df):

        # UDF for lag transformation
        def lag_feature(df):
            df['sales'] = df['sales'].shift(self.offset)
            return df

        self.schema = getSchema(df)
        # applying lag transformation on the bases of store and department
        df = df.groupby('store_id', 'dept_id').applyInPandas(lag_feature, schema=self.schema)
        return df

