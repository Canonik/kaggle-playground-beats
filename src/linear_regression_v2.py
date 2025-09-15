import numpy as np
import random
np.random.seed(42)
random.seed(42)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import standard_training_set, standard_test_set
from src.pipelines import LogTransformer, RatioTransformer

num_att = ["MoodScore", "TrackDurationMs", "RhythmScore",  "InstrumentalScore", "Energy", "AudioLoudness", "VocalContent", "AcousticQuality","LivePerformanceLikelihood",]
log_att = ["RhythmScore","MoodScore","AudioLoudness", "VocalContent", "AcousticQuality", "InstrumentalScore","LivePerformanceLikelihood",]
ratio_att = ["Energy", "AcousticQuality", "VocalContent", "InstrumentalScore"]


clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("BeatsPerMinute", axis=1)
clients_labels = clients["BeatsPerMinute"]

clients_test = standard_test_set()

num_pipeline = make_pipeline(
    PolynomialFeatures(degree=5, interaction_only=False, include_bias=True),
    SimpleImputer(strategy="median"),
    StandardScaler())

log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())

ratio_pipeline = make_pipeline(
    RatioTransformer(),
    StandardScaler())

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("log", log_pipeline, log_att),
    ("rat", ratio_pipeline, ratio_att)
], remainder="drop")



full_pipeline = Pipeline([
    ("full_preprocessing", preprocessing),
    ("model", LinearRegression())])
     
def train_evaluate():
    print("FITTING MODEL")
    tree_model = full_pipeline.fit(clients_attr, clients_labels)
    print("MODEL FIT")

    
    predictions = tree_model.predict(clients_test)

    predictions_df = pd.DataFrame(predictions, columns=["BeatsPerMinute"], index=clients_test.index)

    predictions_df.to_csv("reports/linear_regression_1.csv")

if __name__ == "__main__":
    train_evaluate()
 