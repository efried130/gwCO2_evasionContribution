"""
model.py

Train the XGBoost groundwater CO2 model and run the single-feature ablation
(Supporting Information Table S2).

The training and ablation logic lives in the accompanying modules:
    train_pipeline_aq_uncertainty2.py   (train, CONFIG, NUMERIC_FEATURES)
    ablation.py                         (run_ablation)

This script is the orchestration entry point: it loads df_cleaned, calls
train(), and reproduces the ablation from a saved run.
"""

import os
import joblib
import pandas as pd

from train_pipeline_aq_uncertainty2 import train, CONFIG
import train_pipeline_aq_uncertainty2 as pipeline_module
from ablation import run_ablation

DF_CLEANED = "data/df_cleaned.csv"

# Canonical model run used in the manuscript
MODEL_RUN = "model_runs/xgBoost_CO2aq_SLIM_min2_someGWScapes_simpleConformal_20260505_173720"
EXTRA_CATEGORICALS = ["HYGEO2", "lit_cl_smj"]
SKIP_CV = True


# --- Train ---
df_cleaned = pd.read_csv(DF_CLEANED)
pipeline, save_dirs = train(df_cleaned, CONFIG)


# --- Ablation (SI Table S2) ---


#  Load saved config from model run
saved = joblib.load(os.path.join(MODEL_RUN, 'models', 'complete_pipeline.pkl'))

# Override the module's defaults with the saved run's config/features
pipeline_module.CONFIG = saved['config']
pipeline_module.NUMERIC_FEATURES = saved['numeric_features']

if SKIP_CV:
    pipeline_module.CONFIG['perform_cv'] = False

#  Build ablation dict
ablations = {f'no_{f}': [f] for f in saved['numeric_features']}
for cat in EXTRA_CATEGORICALS:
    ablations[f'no_{cat}'] = [cat]

#  Run
results = run_ablation(df_cleaned, ablations)
