from pyspark.sql import SparkSession
from pyspark.ml import Transformer
import pyspark.sql.functions as f
import warnings
import traceback
warnings.filterwarnings('ignore')
spark = SparkSession.builder.master("local[5]").appName('MLE Assignment').getOrCreate()


class MarkZeroNegTransformer(Transformer):

    def __init__(self):
        super().__init__()
        self.aggregated_file_path = 'aggregated/mark_neg_zero.csv'
        pass

    def _transform(self, df):
        """
        Create new flag column that will show either each store contain any zero or negative sales or not
        If a contain zero/negative sales flag will be 1 to related store else flag will be zero
        :param df:
        :return df:
        """
        try:
            df = df.withColumn('flag', f.when(f.col('sales') <= 0.0, 1).otherwise(0))

            # ignore this line
            self.aggregated_file_path = ""
            return df
        except (Exception,):
            traceback.print_exc()
            return False
