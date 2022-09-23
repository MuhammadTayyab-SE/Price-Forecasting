import pyspark.sql.functions as F
from pyspark.ml import Transformer
import warnings
import os
import traceback
import re
warnings.filterwarnings('ignore')

def get_orignal_columns(lst=list()):
    new_list = list()
    for element in lst:
        try:
            new_list.append(re.sub(r'\W+', ' ', element).strip().split(' ')[1])
        except:
            new_list.append(element)
    return new_list

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
            aggregated_df = df.groupBy(["store_id", 'dept_id', 'date']).avg().orderBy(['store_id', 'dept_id', 'date'])

            # renaming column name according to given data
            new_cols = get_orignal_columns(aggregated_df.columns)

            # change aggregated dataframe column name
            aggregated_df = aggregated_df.toDF(*new_cols)
            aggregated_df = aggregated_df.withColumn('sales', (F.col('demand') * F.col('sell_price')))

            # save_aggregated_data(df=aggregated_df, path=self.aggregated_file_path)
            return aggregated_df
        except:
            traceback.print_exc()
            return None
