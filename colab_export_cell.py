# ==========================================================================
# CELL 20 | Export everything the Streamlit dashboard needs
# Paste this as a NEW cell at the END of your notebook (after Cell 19) and run it.
# It downloads dashboard_upload.zip containing:
#   models/drought_model.pkl, models/metadata.json, data/dashboard_data.csv, requirements.txt
# ==========================================================================
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import sklearn
from google.colab import files

EXPORT_DIR = Path("dashboard_upload")
if EXPORT_DIR.exists():
    shutil.rmtree(EXPORT_DIR)
(EXPORT_DIR / "models").mkdir(parents=True)
(EXPORT_DIR / "data").mkdir(parents=True)


class NumpyEncoder(json.JSONEncoder):
    """Lets json.dump handle numpy numbers and arrays."""
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)


# 1) Model (the best model trained on all data in Cell 19)
joblib.dump(final_model, EXPORT_DIR / "models" / "drought_model.pkl", compress=3)

# 2) Training summary shown on the "Model Performance" page
with open(EXPORT_DIR / "models" / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, cls=NumpyEncoder)

# 3) Clean data used for charts, filters and crop recommendations
app_columns = ["Region", "Location", "Agro_Ecological_Zone", *RAW_NUMERICAL_FEATURES,
               "Drought_Risk", "Recommended_Crops", "Recommended_Actions"]
df[app_columns].to_csv(EXPORT_DIR / "data" / "dashboard_data.csv", index=False)

# 4) requirements.txt with the SAME scikit-learn / xgboost versions used for training
lines = ["streamlit>=1.50", "pandas", "numpy", "plotly>=5.20", "joblib",
         f"scikit-learn=={sklearn.__version__}"]
if best_model_name == "XGBoost":
    import xgboost
    lines.append(f"xgboost=={xgboost.__version__}")
(EXPORT_DIR / "requirements.txt").write_text("\n".join(lines) + "\n")

print("Best model:", best_model_name)
print((EXPORT_DIR / "requirements.txt").read_text())
shutil.make_archive("dashboard_upload", "zip", EXPORT_DIR)
files.download("dashboard_upload.zip")
