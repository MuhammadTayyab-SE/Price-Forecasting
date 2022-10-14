from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.feature import StringIndexer
import warnings
warnings.filterwarnings('ignore')


def get_col_list(cols_list):
    lst = list()
    for i in range(len(cols_list)):
        lst.append(f"{i+1}")
    return lst


class StringIndexerTransformer(Transformer, HasInputCols, HasOutputCols):
    """
    This class is used to convert the
    """
    def __init__(self, input_cols=list, output_cols=list, original='keeps'):
        super().__init__()
        self.inputCols = input_cols
        self.outputCols = output_cols
        self.original = original

    def _transform(self, df):

        string_indexer = StringIndexer(inputCols=self.inputCols, outputCols=get_col_list(self.outputCols))
        converter = string_indexer.fit(df)
        results = converter.transform(df)

        if self.original == 'keeps':
            results = results.drop(*self.outputCols)
            for i in range(len(self.inputCols)):
                results = results.withColumnRenamed(f"{i+1}", self.inputCols[i])
        else:
            for i in range(len(self.inputCols)):
                results = results.withColumnRenamed(f"{i+1}", self.outputCols[i])

        return results

