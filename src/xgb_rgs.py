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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform, loguniform

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
    ("model", XGBRegressor(
    eval_metric="rmse",
    random_state=1234,
    verbosity=2,
    tree_method="auto",
    objective="reg:tweedie"
    ))          
])

param_distributions = {
    "model__n_estimators": randint(150, 5000),       
    "model__learning_rate": uniform(0.001, 0.1),      
    "model__max_depth": randint(4, 19),              
    "model__min_child_weight": randint(1, 19),       
    "model__subsample": uniform(0.6, 0.4),          
    "model__colsample_bytree": uniform(0.6, 0.5),   
    "model__gamma": uniform(0, 0.5),                  
    "model__reg_alpha": uniform(0, 0.5),               
    "model__reg_lambda": uniform(0, 2),
    "model__tweedie_variance_power": uniform(1, 0.5)              
}


if __name__ == "__main__":
    cv = KFold(n_splits=5, shuffle=True, random_state=1234)

    random_search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distributions,
    n_iter=5,                    
    scoring="neg_root_mean_squared_error",
    cv=cv,
    verbose=2,
    random_state=42,
)

    random_search.fit(clients_attr, clients_labels)
    print("Best score:", -random_search.best_score_)
    print("Best params:", random_search.best_params_)
