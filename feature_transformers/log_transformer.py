import pyspark.sql.functions as f
from pyspark.ml import Transformer
import warnings
warnings.filterwarnings('ignore')


class LogTransformer(Transformer):
    def __init__(self):
        super().__init__()
        pass

    def _transform(self, df):
        # Ignore this line
        self.none = None

        df = df.withColumn('sales', f.log10(f.col('sales')))
        return df
