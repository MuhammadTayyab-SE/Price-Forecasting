import pyspark.sql.functions as F
from pyspark.ml import Transformer
import warnings
import os
import traceback
import re
warnings.filterwarnings('ignore')

# remove unnecessary or special chars from the string
def get_orignal_columns(lst=list()):
    """
    remove special charaters form the string
    for example we have some string like [avg(count), avg(product)] this function will return [count, product]
    :param list:
    :return list:
    """
    new_list = list()
    for element in lst:
        try:
            new_list.append(re.sub(r'\W+', ' ', element).strip().split(' ')[1])
        except:
            new_list.append(element)
    return new_list

# return schema of that dataframe in string format
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



class AggregatedTransformer(Transformer):
    def __init__(self):
        self.aggregated_file_path = 'aggregated/aggregated_data.csv'


    def _transform(self, df):
        try:
            # grouped data on store-dept level
            aggregated_df = df.groupBy(["store_id", 'dept_id', 'date']).avg().orderBy(['store_id', 'dept_id', 'date'])

            # renaming column name according to given data
            new_cols = get_orignal_columns(aggregated_df.columns)

            # change aggregated dataframe column name
            aggregated_df = aggregated_df.toDF(*new_cols)

            # add new column named sales that is the product of demand and sell_price
            aggregated_df = aggregated_df.withColumn('sales', (F.col('demand') * F.col('sell_price')))

            return aggregated_df
        except:
            traceback.print_exc()
            return None
