import joblib
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data_loading import standard_training_set, standard_test_set, full_original_database
from src.pipelines import day_month_encoding, log_encoding, month_sin_encoding, month_cos_encoding, LogTransformer, DayOfYearTransformer, RF_useless_features, WrapperWithY, OOFJobTransformer

num_att = ["age", "pdays", "previous", "duration"] 
cat_att =[ "job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"] #-job
log_att =["balance", "campaign", "duration"] 
time_att = ["day", "month"]
job_att = ["job"]
useless_att = ["previous", "age","balance","duration","campaign"]
cyclical_att = ["month_sin", "month_cos", "day_sin", "day_cos"]


clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("y", axis=1)
clients_labels = clients["y"]

full_data = full_original_database().reset_index(drop=True)
full_data_clients_attr = full_data.drop("y", axis=1)
full_data_clients_labels = full_data["y"]

clients_attr = pd.concat([clients_attr, full_data_clients_attr], ignore_index=True)
clients_labels = pd.concat(
    [clients_labels.reset_index(drop=True), full_data_clients_labels.reset_index(drop=True)],
    ignore_index=True
)
clients_test = standard_test_set()


num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False))

log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())

day_pipeline = make_pipeline(
    DayOfYearTransformer(),
    StandardScaler()
)
job_pipeline = make_pipeline(
    OOFJobTransformer(columns=cat_att + job_att)
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("log", log_pipeline, log_att),
    ("cat", cat_pipeline, cat_att),
    ("day", day_pipeline, time_att),
], remainder="drop")

oof_preprocessing = ColumnTransformer([
    ("oof", job_pipeline, cat_att)
], remainder="drop")

full_preprocessing = FeatureUnion([
    ("oof_preprocessing", oof_preprocessing),
    ("preprocessing", preprocessing)
])

full_pipeline = Pipeline([
    ("full_preprocessing", full_preprocessing),
    ("model", XGBClassifier(
    colsample_bytree=0.4,        
    gamma=0.0,                 
    learning_rate=0.005,      
    max_depth=20,                
    min_child_weight=1,          
    n_estimators=20000,          
    reg_alpha=0.0,             
    reg_lambda=0.0,            
    scale_pos_weight=1.5883,     
    subsample=0.7,             
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    verbosity=1,
    tree_method="hist",          
    grow_policy="lossguide",     
))
])

if __name__ == "__main__":
#CV AUC: 0.96725 job + oof job encoding
#CV AUC: 0.96725 oof cat + cat 
#CV AUC: 0.96724 simple pipeline, no oof encoding
#CV AUC: 0.96723 oof encoding
#CV AUC: 0.96672 oof cat encoding    
    ''' #cross eval auc with 5 folds
    scores = cross_val_score(
        full_pipeline, clients_attr, clients_labels, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1, 
        verbose=2,
        scoring="roc_auc")
    
    print(f"CV AUC: {scores.mean():.5f}")
    
    '''
    #full pipeline on full training set
    XGB_model = full_pipeline.fit(clients_attr, clients_labels)
    
    joblib.dump(XGB_model, "models/XGB_finetuned_overfit_beast.pkl")
    
    predictions = XGB_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/XGB_finetuned_overfit_beast.csv")
    '''
    predictions_df = pd.DataFrame(np.zeros((len(clients_attr), 1)), columns=["y"], index=clients_attr.index)

    for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(clients_attr, clients_labels):
        X_train_fold = clients_attr.iloc[train_idx]
        X_val_fold = clients_attr.iloc[val_idx]
        y_train_fold = clients_labels.iloc[train_idx]
        y_val_fold = clients_labels.iloc[val_idx]


        xgb_model = full_pipeline.fit(X_train_fold, y_train_fold)
        predictions_df.iloc[val_idx, 0] = xgb_model.predict_proba(X_val_fold)[:, 1]

    predictions_df.to_csv("reports/XGB_finetuned_30h_pipeline_cv5.csv")
    '''