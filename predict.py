"""
predict.py

Run the trained pipeline over the gridded CONUS predictor stack to produce
monthly 0.1-degree groundwater CO2(aq) predictions with conformal uncertainty.

Prediction logic lives in:
    predict_gridded_aq_uncertainty2.py   (predict_gridded)
"""

import sys
import joblib

sys.path.insert(0, "model_scripts")
from predict_gridded_aq_uncertainty2 import predict_gridded

PIPELINE_PATH    = "model_runs/xgBoost_CO2aq_SLIM_min2_someGWScapes_simpleConformal_20260505_173720/models/complete_pipeline.pkl"
ERA5_DAILY_DIR   = "data/era5_parquet_01deg"
ERA5_MONTHLY_DIR = "data/era5_month_parquet_01deg"
GLDAS_DIR        = "data/gldas_parquet_01deg"
HYDRO_DIR        = "data/hydroatlas_parquet_01deg"
OUT_DIR          = "data/co2_predictions_aq_uncertainty"

YEAR_RANGE = (1970, 2024)


pipeline = joblib.load(PIPELINE_PATH)

predict_gridded(
    pipeline=pipeline,
    era5_daily_dir=ERA5_DAILY_DIR,
    era5_month_dir=ERA5_MONTHLY_DIR,
    hydro_dir=HYDRO_DIR,
    gldas_dir=GLDAS_DIR,
    out_dir=OUT_DIR,
    year_range=YEAR_RANGE,
    predict_uncertainty=True,
    batch_log_every=1,
)
