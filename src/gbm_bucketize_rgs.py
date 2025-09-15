import numpy as np
import random
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV, train_test_split
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
bins_train = np.linspace(clients_labels.min(), clients_labels.max(), 70)

clients_labels = pd.cut(clients_labels, bins=bins_train, labels=False, include_lowest=True)


num_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, interaction_only=False, include_bias=False),
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
        random_state=4321,
        )
    )
])

param_distributions = {
    "model__learning_rate": uniform(0.01, 0.045),
    "model__n_estimators": randint(200, 350),
    "model__num_leaves": randint(20, 50),
    "model__min_child_samples": randint(30, 45),
    "model__reg_alpha": uniform(1.5, 5),
    "model__reg_lambda": uniform(4, 5),
    "model__max_bin": randint(180, 255),
    "model__subsample": uniform(0.85, 0.15),
    "model__colsample_bytree": uniform(0.9, 0.1),
    "model__class_weight": ["balanced", None]
}



cv = KFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_distributions,
    n_iter=35,                 
    cv=cv,
    scoring="neg_root_mean_squared_error",
    verbose=2,
    random_state=4321,              
)

random_search.fit(clients_attr, clients_labels)
print("best score (cv):", -random_search.best_score_)
print("best params:", random_search.best_params_)

best = pd.DataFrame(random_search.best_params_, columns= list(random_search.best_params_.keys()), index=range(1))
best.to_csv("reports/bucket_gbm_best_params_3")