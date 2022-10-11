from pyspark.ml import Estimator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pyspark.ml.param.shared import *
import warnings
import hyperopt
from hyperopt import fmin, tpe, hp, Trials
from pyspark.ml.classification import RandomForestClassifier
warnings.filterwarnings('ignore')

class RandomForestEstimator(Estimator, HasInputCol, HasOutputCols):
    def __int__(self):
        super(RandomForestEstimator, self).__int__()
        self.model = RandomForestClassifier(labelCol='sales', featuresCol='features')