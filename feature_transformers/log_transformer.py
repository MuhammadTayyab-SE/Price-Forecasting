import pyspark.sql.functions as f
from pyspark.ml import Transformer
import warnings
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
warnings.filterwarnings('ignore')


class LogTransformer(Transformer, HasInputCols, HasOutputCols):
    def __init__(self, column: list):
        super().__init__()
        self.columns = column

    def _transform(self, df):
        """
        Gets a pyspark dataframe and return dataframe with log transformation on given columns
        :param df: dataframe
        :return: dataframe with log transformed columns
        """
        column_list = [x[0] for x in df.dtypes]

        for column in self.columns:
            if column in column_list:
                df = df.withColumn(column, f.log10(f.col(column)))
            else:
                print(f"{column} not in the list of columns {column_list}")
        return df
