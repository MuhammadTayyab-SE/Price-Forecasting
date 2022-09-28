from pyspark.ml import Transformer
import warnings
warnings.filterwarnings('ignore')


# return schema of the dataframe in string format
def get_schema(df):
    """ This function takes dataframe as argument and return dataframe's schema as string
    :param df: dataframe
    :return: df
    """
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
        super().__init__()
        self.offset = offset
        self.schema = str()

    def _transform(self, df):

        # UDF for lag transformation
        def lag_feature(dff):
            """ This function apply lag transformation on each of the grouped data
            :param dff: dataframe
            :return: dataframe
            """
            dff['sales'] = dff['sales'].shift(self.offset)
            return dff

        self.schema = get_schema(df)
        # applying lag transformation on the bases of store and department
        df = df.groupby('store_id', 'dept_id').applyInPandas(lag_feature, schema=self.schema)
        return df
