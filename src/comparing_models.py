import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import standard_training_set, standard_test_set, full_original_database

#XGB_finetuned_30h_pipeline.csv
#XGB_blended_submission11.csv
#model_stacking_xgb.csv
#reports/XGB_best_oof_cat_plus_cat.csv
#XGB_finetuned_overfit_beast.csv
#rfc_blending_weak.csv
#kneighbor_blending_weak.csv
#"reports/rf_poly_pipeline.csv"
#CatBoost_maxxing1.csv"

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

preds1 = pd.read_csv("reports/XGB_0.csv") 
preds2 = pd.read_csv("reports/XGB_1.csv")
preds3 = pd.read_csv("reports/XGB_2.csv")       


assert all(preds1["id"] == preds2["id"]), "IDs do not match!"
assert all(preds2["id"] == preds3["id"]), "IDs do not match!"

# Extract y values and reshape
p_1 = preds1["BeatsPerMinute"].values.reshape(-1, 1)
p_2 = preds2["BeatsPerMinute"].values.reshape(-1, 1)
p_3 = preds3["BeatsPerMinute"].values.reshape(-1, 1)

# Pearson correlations & mean diffs (optional, for info)
corr_2 = np.corrcoef(p_1.flatten(), p_2.flatten())[0, 1]
diff_2 = (p_1.flatten() - p_2.flatten()).mean()
corr_3 = np.corrcoef(p_3.flatten(), p_1.flatten())[0, 1]
diff_3 = (p_3.flatten() - p_1.flatten()).mean()

print(f"Second Pearson correlation with XGB: {corr_2:.6f}, mean diff: {diff_2:.6f}")
print(f"Third Pearson correlation with XGB: {corr_3:.6f}, mean diff: {diff_3:.6f}")


# Set blend weights - sum to 1.0
w_xgb = 0.87
w_2 = 0.10
w_3 = 0.03

# Blend predictions
blended_y = w_xgb * p_1 + w_2 * p_2+ w_3 * p_3

# Save submission
blended = pd.DataFrame({
    "id": preds1["id"],
    "BeatsPerMinute": blended_y
})
blended.to_csv("reports/blended_submission0.csv", index=False)
print("Blended submission saved.")