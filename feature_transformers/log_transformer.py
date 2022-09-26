from pyspark.sql.session import SparkSession
import pyspark.sql.functions as F
import numpy as np
from pyspark.ml import Transformer
import warnings
warnings.filterwarnings('ignore')


class LogTransformer(Transformer):
    def __init__(self):
        pass

    def _transform(self, df):
        df = df.withColumn('sales', F.log10(F.col('sales')))
        return df
