import pyspark.sql.functions as F
from pyspark.ml import Transformer
import warnings
import os
import traceback
warnings.filterwarnings('ignore')


def get_integer_columns(df):
    integer_cols = [col for col, dtype in df.dtypes if dtype == 'int']
    return integer_cols


def cast_df_col_to_int(aggregated_df):
    # change the datatype of column form long/double to int
    aggregated_df = aggregated_df.select(
        [F.col(column).cast('int') if dtype != 'string' else F.col(column).cast(dtype) for column, dtype in
         aggregated_df.dtypes])
    return aggregated_df


def save_aggregated_data(df, path):
    try:
        if os.path.isfile(path):
            # if file exist remove that file
            os.remove(path)
        df = df.toPandas()
        df.to_csv(path, index=False)
        return True
    except:
        traceback.print_exc()
        return False


class AggregatedTransformer(Transformer):

    def __init__(self):
        self.aggregated_file_path = 'aggregated/aggregated_data.csv'


    def _transform(self, df):
        try:
            aggregated_df = df.groupBy(["store_id", 'dept_id']).sum().orderBy(['store_id', 'dept_id'])

            # renaming column name according to given data
            new_cols = list()
            new_cols.append(aggregated_df.columns[0])
            new_cols.append(aggregated_df.columns[1])
            new_cols.extend(get_integer_columns(df))

            # change aggregated dataframe column name
            aggregated_df = aggregated_df.toDF(*new_cols)
            aggregated_df = cast_df_col_to_int(aggregated_df)
            # save_aggregated_data(df=aggregated_df, path=self.aggregated_file_path)
            return aggregated_df
        except:
            traceback.print_exc()
            return None
