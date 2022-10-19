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

    def __init__(self, offset=int(), lagged_column='sales'):
        super().__init__()
        self.offset = offset
        self.schema = str()
        self.lagged_column = lagged_column

    def _transform(self, df):

        # UDF for lag transformation
        def lag_feature(dff):
            """ This function apply lag transformation on each of the grouped data
            :param dff: dataframe
            :return: dataframe
            """
            for itr in range(0, self.offset):
                col_name = f"sales_lag_{itr + 1}"
                dff[col_name] = dff[self.lagged_column].copy().shift(itr+1)

            return dff

        self.schema = get_schema(df)

        # appending schema for lag transformed columns
        self.schema += ', '
        string = ""
        for i in range(0, self.offset):
            if i == self.offset - 1:
                string += f"{self.lagged_column}_lag_{i + 1} double"
            else:
                string += f"{self.lagged_column}_lag_{i+1} double, "
        self.schema += string

        # applying lag transformation on the bases of store and department
        df = df.groupby('store_id', 'dept_id').applyInPandas(lag_feature, schema=self.schema)
        return df
