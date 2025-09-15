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

# bucketize target to avoid model adapting to mean
bins_train = np.linspace(clients_labels.min(), clients_labels.max(), 13)

clients_labels = pd.cut(clients_labels, bins=bins_train, labels=False, include_lowest=True)

clients_test = standard_test_set()

num_pipeline = make_pipeline(
    PolynomialFeatures(degree=4, interaction_only=False, include_bias=True),
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
    ("model", LGBMClassifier(
    class_weight=None,
    colsample_bytree=0.9348920936908014,
    learning_rate=0.020112732286051465,
    max_bin=181,
    min_child_samples=44,
    n_estimators=269,
    num_leaves=35,
    reg_alpha=3.4883053854301567,
    reg_lambda=7.873993683489575,
    subsample=0.9365104689993639,
    ))
])
#26.4616955916462

def train_evaluate():
    print("FITTING MODEL")
    tree_model = full_pipeline.fit(clients_attr, clients_labels)
    print("MODEL FIT")
    
    prediction_buckets = tree_model.predict(clients_test)

    bins_values = (bins_train[1:] + bins_train[:-1]) / 2
    predictions = bins_values[prediction_buckets]

    predictions_df = pd.DataFrame(predictions, columns=["BeatsPerMinute"], index=clients_test.index)

    predictions_df.to_csv("reports/bucket_gbm_2.csv")

def kf_evaluate():
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    scores = cross_val_score(full_pipeline, clients_attr, clients_labels, cv=kf, scoring='neg_root_mean_squared_error')
    print(-scores.mean())



if __name__ == "__main__":
    train_evaluate()