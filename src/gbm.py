import joblib
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
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform


import datetime

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
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
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
    ("model", LGBMRegressor(
        colsample_bytree=0.7073899427560627,
        learning_rate=0.003326399371381578,
        max_bin=71,
        min_child_samples=29,
        n_estimators=865,
        num_leaves=17,
        reg_alpha=4.42036702053619,
        reg_lambda=0.24294123442692134,
        subsample=0.8460028906796679
    ))
])
#26.4616955916462

def train_evaluate():
    print("FITTING MODEL")
    tree_model = full_pipeline.fit(clients_attr, clients_labels)
    print("MODEL FIT")
    #joblib.dump(XGB_model, "models/XGB_2.pkl")
    
    predictions = tree_model.predict(clients_test)

    predictions_df = pd.DataFrame(predictions, columns=["BeatsPerMinute"], index=clients_test.index)

    predictions_df.to_csv("reports/gbm_4.csv")

def kf_evaluate():
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    scores = cross_val_score(full_pipeline, clients_attr, clients_labels, cv=kf, scoring='neg_root_mean_squared_error')
    print(-scores.mean())



if __name__ == "__main__":
    train_evaluate()
