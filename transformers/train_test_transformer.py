import pyspark.sql.functions as F
from pyspark.ml import Transformer
from .aggregated_transformer import get_integer_columns, cast_df_col_to_int, save_aggregated_data
import warnings
warnings.filterwarnings('ignore')


class TrainTestTransformer(Transformer):

    def __init__(self):
        self.aggregated_file_path = 'aggregated/mark_neg_zero.csv'

    def _transform(self, df):
        params = {
            'train': 0.7,
            'test': 0.3
        }
        if len(params) == 2:
            df_cols = get_integer_columns(df)

            # getting common columns
            common_cols = list(set(df.columns) - set(df_cols))
            split_cols = list(set(df_cols) - set(common_cols))

            split_cols = df_cols[:len(df_cols)]

            train_ratio = int(len(split_cols) * params['train'])
            test_ratio = len(split_cols) - train_ratio

            train_cols = common_cols.copy()
            train_cols.extend(split_cols[:train_ratio + 1])

            test_cols = common_cols.copy()
            test_cols.extend(split_cols[train_ratio + 1:])

            train_df = df.select(train_cols)
            test_df = df.select(test_cols)

            return train_df, test_df
        else:
            print("Please enter split ratios")
            return None
