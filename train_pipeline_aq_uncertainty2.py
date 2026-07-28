"""
train_pipeline.py
=================
XGBoost CO2 prediction training with Split Conformal Prediction
uncertainty quantification.

Key features:
  1. All transform logic in shared functions
  2. Pipeline artifact saved with EVERYTHING needed for inference + uncertainty
  3. Split conformal prediction: constant-width intervals in log-space with
     formal coverage guarantees. Simple, honest, and well-calibrated.
  4. RFECV included
  5. CV captures fold-level feature importances + well-level stability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import RFECV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
from scipy.stats import gaussian_kde, ks_2samp
from category_encoders import TargetEncoder
from xgboost import XGBRegressor
import joblib
import warnings
import os
from datetime import datetime
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # --- Thresholds ---
    'label': 'CO2_aq',
    'label_units': 'mol/L',
    'threshold_co2': 0.00002,
    'threshold_co2_high': 0.006,
    'analyze_test_distributions': False,

    # --- Dedup / filtering ---
    'perform_deduplication': False,
    'dedup_by_well': True,
    'well_column': 'MonitoringLocation',
    'filter_by_location_count': True,
    'min_samples_per_location': 2,

    # --- NA / outlier ---
    'remove_rows_with_all_na': True,
    'remove_rows_with_most_na': True,
    'most_na_threshold': 0.7,
    'na_removal_threshold': 0.4,
    'outlier_removal_method': 'iqr',
    'outlier_threshold': 3.0,
    'min_features_for_outlier': 5,

    # --- Target ---
    'eliminate_outside_range': False,
    'target_scale_min': -2,
    'target_scale_max': 2,

    # --- Categoricals ---
    'categorical_features': ['lit_cl_smj', 'HYGEO2'],
    'encoding_strategy': {'HYGEO2': 'target',
                          'cls_cl_smj': 'target',
                          'lit_cl_smj': 'target',
                          'tec_cl_smj': 'target'},

    # --- Model ---
    'show_scaled_distributions': True,
    'test_size': 0.2,
    'random_state': 1,
    'perform_cv': True,
    'cv_folds': 10,
    'save_directory': '/nas/cee-water/cjgleason/ellie/co2/src/model_runs',
    'save_dataframes': True,
    'save_plots': True,
    'xgb_params': {
        'n_estimators': 900,
        'max_depth': 5,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'colsample_bynode': 0.9,
        'max_cat_to_onehot': 4,
        'random_state': 42,
        'n_jobs': -1,
    },
    'run_name': 'xgBoost_CO2aq_SLIM_min2_someGWScapes_simpleConformal_noHyporheicSiteType',

    # ── UQ: Simple conformal prediction ───────────────────────────
    'conformal_levels': [0.68, 0.90, 0.95],
    'conformal_calib_frac': 0.25,
}

NUMERIC_FEATURES = [
    'year', 'season', 'cly_pc_sav', 'soc_th_sav', 'dtb',
    'swvl1_month', 'tp_month',
    'ele_mt_sav', 'gwt_cm_sav', 'kar_pc_sse', 'slp_dg_sav',
    'arid_organic_index', 'urb_pc_sse', 'crp_pc_sse', 'Dd',
    'swc_proximity_sink', 'lai_hv_month', 'coastal_plain_index',
    'temp_moisture', 'swc_pc_s', 'ari_ix_sav', 'Irrigation',
] # 'Irrigation',


# Display name mapping for feature importance plot
FEATURE_DISPLAY_NAMES = {
    'arid_organic_index':   'Aridity x SOC',
    'swc_pc_s':             'Soil Water (%)',
    'HYGEO2_target':        'Recharge (cat)',
    'ele_mt_sav':           'Elevation (avg, m)',
    'lit_cl_smj_target':    'Lithology (cat)',
    'urb_pc_sse':           'Urban Fraction (%)',
    'cly_pc_sav':           'Clay Fraction (%)',
    'kar_pc_sse':           'Karst Fraction (%)',
    'coastal_plain_index':  'Coastal Plain Index',
    'ari_ix_sav':           'Aridity Index',
    'Dd':                   'Drainage Density',
    'slp_dg_sav':           'Slope (avg, m/m)',
    'Irrigation':           'Irrigation Index',
    'soc_th_sav':           r'Soil Organic Carbon (t ha$^{-1}$)',
    'dtb':                  'Depth to Bedrock',
}


INTERACTION_DEFS = [
    ('aridity_precip',        'ari_ix_sav',  'tp_month',          'multiply'),
    ('temp_moisture',         't2m_month',         'swc_pc_s',    'multiply'),
    ('evap_precip_product',   'evavt_month',       'tp_month',          'multiply'),
    ('elevation_erosion',     'ele_mt_sav',  'ero_kh_sav',  'multiply'),
    ('elevation_soil_carbon', 'ele_mt_sav',  'soc_th_sav',  'multiply'),
    ('temp_moisture_sq',      'temp_moisture', None,         'square'),
    ('distance_drainage',     'DIST_MAIN',   'Dd',          'multiply'),
    ('depth_weighted_soc',    'soc_th_sav',  'dtb',         'multiply'),
    ('flushing_index',        'tp_month',    'slp_dg_sav',  'multiply'),
    ('precip_proximity_sink', 'tp_month',    'DIST_SINK',   'multiply'),
    ('shallow_organic_index', 'soc_th_sav',  'dtb',         'ratio'),
    ('swc_proximity_sink',    'swc_pc_s',    'DIST_SINK',   'multiply'),
    ('hydraulic_gradient',    'slp_dg_sav',  'UP_AREA',     'multiply'),
    ('arid_organic_index', 'soc_th_sav', 'ari_ix_sav', 'multiply'),
    ('coastal_plain_index', 'ele_mt_sav', 'DIST_SINK', 'ratio'),
]


# ============================================================================
# SHARED TRANSFORM FUNCTIONS  (used by both train and predict)
# ============================================================================

def create_interaction_terms(df, interaction_defs=INTERACTION_DEFS):
    df = df.copy()
    for out_col, col_a, col_b, op in interaction_defs:
        if op == 'multiply':
            if col_a in df.columns and col_b in df.columns:
                df[out_col] = df[col_a] * df[col_b]
            else:
                df[out_col] = np.nan
        elif op == 'ratio':
            if col_a in df.columns and col_b in df.columns:
                denom = df[col_b].clip(lower=10)
                df[out_col] = df[col_a] / denom
            else:
                df[out_col] = np.nan
        elif op == 'square':
            if col_a in df.columns:
                df[out_col] = df[col_a] ** 2
            else:
                df[out_col] = np.nan
        elif op == 'normalize':
            if col_a in df.columns and col_b in df.columns:
                df[out_col] = df[col_b] / df[col_a].clip(lower=1)
            else:
                df[out_col] = np.nan

    return df


def add_temporal_features(df, date_col='Date'):
    df = df.copy()
    if date_col in df.columns:
        df[date_col]      = pd.to_datetime(df[date_col])
        df['month']       = df[date_col].dt.month
        df['season']      = df[date_col].dt.month % 12 // 3 + 1
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['year']        = df[date_col].dt.year
        df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin']     = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos']     = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    return df


def encode_categoricals(df_cat, encoders, config, y_train=None, fit=False):
    if df_cat.empty:
        return np.array([]).reshape(len(df_cat), 0), encoders, []

    cat_cols = list(df_cat.columns)
    strategy = config['encoding_strategy']

    df_filled = df_cat.copy()
    for col in cat_cols:
        df_filled[col] = df_filled[col].astype(str).replace(
            ['nan', 'None', 'NaN', '<NA>'], 'Missing')

    ordinal_cols = [c for c in cat_cols if strategy.get(c) == 'ordinal']
    target_cols  = [c for c in cat_cols if strategy.get(c) == 'target']
    onehot_cols  = [c for c in cat_cols if strategy.get(c) == 'onehot']

    encoded_arrays = []
    feature_names  = []

    for col in ordinal_cols:
        if fit:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=-1, encoded_missing_value=-2)
            encoded_arrays.append(enc.fit_transform(df_filled[[col]]))
            encoders[col] = enc
        else:
            encoded_arrays.append(encoders[col].transform(df_filled[[col]]))
        feature_names.append(f"{col}_encoded")

    if target_cols:
        if fit:
            enc = TargetEncoder(cols=target_cols, smoothing=25.0,
                                min_samples_leaf=5, return_df=True)
            encoded_arrays.append(enc.fit_transform(df_filled[target_cols], y_train).values)
            encoders['target'] = enc
        else:
            encoded_arrays.append(encoders['target'].transform(df_filled[target_cols]).values)
        feature_names.extend([f"{c}_target" for c in target_cols])

    if onehot_cols:
        if fit:
            enc = OneHotEncoder(drop='first', sparse_output=False,
                                handle_unknown='ignore', dtype=np.float32)
            encoded_arrays.append(enc.fit_transform(df_filled[onehot_cols]))
            encoders['onehot'] = enc
        else:
            encoded_arrays.append(encoders['onehot'].transform(df_filled[onehot_cols]))
        feature_names.extend(list(encoders['onehot'].get_feature_names_out(onehot_cols)))

    if encoded_arrays:
        return np.hstack(encoded_arrays), encoders, feature_names
    return np.array([]).reshape(len(df_cat), 0), encoders, feature_names


def impute_numeric(X_df, imputers, features, fit=False):
    X = X_df.copy()
    high_missing = any(
        (X[f].isnull().sum() / len(X)) * 100 > 50 for f in features if f in X.columns
    )

    if fit:
        imputers = {}
        if high_missing:
            knn = KNNImputer(n_neighbors=5, weights='distance')
            X = pd.DataFrame(knn.fit_transform(X), columns=X.columns)
            imputers['knn'] = knn
        else:
            for feat in features:
                if feat not in X.columns:
                    continue
                missing = X[feat].isnull().sum()
                if missing == 0:
                    continue
                train_data = X[feat].dropna()
                strat = 'median' if abs(stats.skew(train_data)) > 1 else 'mean'
                imp = SimpleImputer(strategy=strat)
                X[[feat]] = imp.fit_transform(X[[feat]])
                imputers[feat] = imp
        return X, imputers

    if 'knn' in imputers:
        X = pd.DataFrame(imputers['knn'].transform(X), columns=X.columns)
    else:
        for feat, imp in imputers.items():
            if feat in X.columns:
                X[[feat]] = imp.transform(X[[feat]])
    return X, imputers


def scale_numeric(X_df, scalers, features, fit=False):
    X = X_df.copy()

    if fit:
        scalers = {}
        for feat in features:
            if feat not in X.columns:
                continue
            train_data = X[feat].values
            skewness   = stats.skew(train_data[~np.isnan(train_data)])
            q25, q75   = np.nanpercentile(train_data, [25, 75])
            iqr        = q75 - q25
            outlier_pct = (
                (train_data < q25 - 1.5 * iqr) | (train_data > q75 + 1.5 * iqr)
            ).sum() / len(train_data) * 100

            use_log, shift = False, 0
            if abs(skewness) > 1.5:
                train_min = np.nanmin(train_data)
                shift = abs(train_min) + 1 if train_min <= 0 else 0
                log_temp = np.log(train_data + shift + 1)
                if abs(stats.skew(log_temp[~np.isnan(log_temp)])) < abs(skewness):
                    use_log = True

            if use_log:
                X_log = np.log(X[feat] + shift + 1).values.reshape(-1, 1)
                q25f, q75f = np.nanpercentile(X_log, [25, 75])
                iqrf = q75f - q25f
                outlier_pct_f = (
                    (X_log.ravel() < q25f - 1.5 * iqrf) |
                    (X_log.ravel() > q75f + 1.5 * iqrf)
                ).sum() / len(X_log) * 100
                scaler = RobustScaler() if outlier_pct_f > 10 else StandardScaler()
                X[[feat]] = scaler.fit_transform(X_log)
                scalers[feat] = {'scaler': scaler, 'shift': shift, 'use_log': True,
                                 'method': f"Log+{'Robust' if outlier_pct_f > 10 else 'Standard'}"}
            else:
                scaler = RobustScaler() if outlier_pct > 10 else StandardScaler()
                X[[feat]] = scaler.fit_transform(X[[feat]])
                scalers[feat] = {'scaler': scaler, 'use_log': False,
                                 'method': 'Robust' if outlier_pct > 10 else 'Standard'}
        return X, scalers

    for feat, info in scalers.items():
        if feat not in X.columns:
            continue
        if info.get('use_log', False):
            shift = info['shift']
            X[feat] = np.log(X[feat] + shift + 1)
        X[[feat]] = info['scaler'].transform(X[[feat]])
    return X, scalers


def prepare_features(df, pipeline, fit=False, y_train=None):
    config   = pipeline['config']
    features = pipeline['numeric_features']

    X_num = df[features].copy()
    X_num, pipeline['imputers'] = impute_numeric(
        X_num, pipeline.get('imputers', {}), features, fit=fit)
    X_num, pipeline['scalers'] = scale_numeric(
        X_num, pipeline.get('scalers', {}), features, fit=fit)

    cat_cols = [c for c in config['categorical_features'] if c in df.columns]
    X_cat_df = df[cat_cols].copy() if cat_cols else pd.DataFrame(index=df.index)
    X_cat, pipeline['encoders'], pipeline['encoded_feature_names'] = encode_categoricals(
        X_cat_df, pipeline.get('encoders', {}), config, y_train=y_train, fit=fit)

    if X_cat.size > 0:
        X_final = np.hstack([X_num.values, X_cat])
    else:
        X_final = X_num.values
    pipeline['all_feature_names'] = features + pipeline['encoded_feature_names']
    return X_final, pipeline


# ============================================================================
# METRICS
# ============================================================================

def calculate_comprehensive_metrics(y_true, y_pred, dataset_name="Test"):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred)
    smpe  = np.mean((y_pred - y_true) / y_true) * 100
    r2    = r2_score(y_true, y_pred)

    y_true_safe = np.clip(y_true, 1e-12, None)
    y_pred_safe = np.clip(y_pred, 1e-12, None)

    rmsle     = np.sqrt(np.mean((np.log(y_pred_safe) - np.log(y_true_safe))**2))
    rel_error = 10**(np.median(np.abs(np.log10(y_pred_safe / y_true_safe)))) - 1

    mean_bias     = np.mean(y_pred - y_true)
    pbias         = 100 * np.sum(y_pred - y_true) / np.sum(y_true)
    log_bias      = 10**(np.mean(np.log10(y_pred_safe / y_true_safe)))
    residuals     = y_pred - y_true
    abs_norm_bias = np.abs(np.mean(residuals)) / np.mean(y_true)

    nse_val = 1 - (np.sum((y_pred - y_true)**2) / np.sum((y_true - np.mean(y_true))**2))

    r_corr  = np.corrcoef(y_true, y_pred)[0, 1]
    alpha   = np.std(y_pred) / np.std(y_true)
    beta    = np.mean(y_pred) / np.mean(y_true)
    kge_val = 1 - np.sqrt((r_corr - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    rrmse = rmse / np.mean(y_true)

    return {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape, 'smpe': smpe,
        'rmsle': rmsle, 'relative_error': rel_error,
        'mean_bias': mean_bias, 'abs_norm_bias': abs_norm_bias,
        'pbias': pbias, 'log_bias': log_bias,
        'nse': nse_val, 'kge': kge_val, 'rrmse': rrmse,
        'n_samples': len(y_true),
    }


def print_comprehensive_metrics(metrics_dict, dataset_name="Test"):
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} SET - COMPREHENSIVE METRICS")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'Value':<15} {'Description':<45}")
    print(f"{'-'*80}")
    rows = [
        ('R²',             'r2',             '.4f', 'Coefficient of determination'),
        ('RMSE',           'rmse',           '.2e', 'Root mean squared error'),
        ('MAE',            'mae',            '.2e', 'Mean absolute error'),
        ('SMPE (%)',       'smpe',           '.2f', 'Signed mean percent error'),
        ('RMSLE',          'rmsle',          '.4f', 'Root mean squared log error'),
        ('RRMSE',          'rrmse',          '.4f', 'Relative RMSE'),
        ('Relative Error', 'relative_error', '.4f', 'Median relative error (multiplicative)'),
        ('Mean Bias',      'mean_bias',      '.2e', 'Mean prediction bias'),
        ('PBIAS (%)',      'pbias',          '.2f', 'Percent bias'),
        ('Log Bias',       'log_bias',       '.4f', 'Geometric mean bias'),
        ('NSE',            'nse',            '.4f', 'Nash-Sutcliffe efficiency'),
        ('KGE',            'kge',            '.4f', 'Kling-Gupta efficiency'),
    ]
    for label, key, fmt, desc in rows:
        print(f"{label:<20} {metrics_dict[key]:<15{fmt}} {desc:<45}")
    print(f"{'n':<20} {metrics_dict['n_samples']:<15,} {'Number of samples':<45}")
    print(f"{'='*80}\n")


# ============================================================================
# WELL-LEVEL METRICS + PLOTTING
# ============================================================================

def calculate_well_level_metrics(results_df, split_name, min_samples=15):
    well_metrics_list = []
    excluded = 0
    for well_id, group in results_df.groupby("MonitoringLocation"):
        y_true = group["CO2_actual"].values
        y_pred = group["CO2_predicted"].values
        if len(y_true) < min_samples:
            excluded += 1
            continue
        m = calculate_comprehensive_metrics(y_true, y_pred)
        m["well_id"]   = well_id
        m["split"]     = split_name
        m["n_samples"] = len(y_true)
        well_metrics_list.append(m)
    df_out = pd.DataFrame(well_metrics_list)
    print(f"\n{split_name.upper()} Well-Level: {results_df['MonitoringLocation'].nunique()} total | "
          f"{len(df_out)} meet n≥{min_samples} | {excluded} excluded")
    return df_out


def compute_aggregate_well_stats(well_metrics_df, split_name):
    if len(well_metrics_df) == 0:
        return None
    metrics_to_agg = ['r2', 'rmse', 'mae', 'smpe', 'rmsle', 'relative_error', 'mape'
                      'nse', 'kge', 'rrmse', 'abs_norm_bias', 'pbias', 'mean_bias']
    agg = {}
    for metric in metrics_to_agg:
        if metric not in well_metrics_df.columns:
            continue
        vals = well_metrics_df[metric].dropna().values
        wts  = well_metrics_df.loc[well_metrics_df[metric].notna(), 'n_samples'].values
        if len(vals) == 0:
            continue
        si  = np.argsort(vals)
        sv  = vals[si]
        sw  = wts[si]
        cs  = np.cumsum(sw)
        wmi = np.searchsorted(cs, cs[-1] / 2.0)
        agg[metric] = {
            'unweighted_mean': np.mean(vals), 'unweighted_median': np.median(vals),
            'unweighted_std': np.std(vals),
            'weighted_mean': np.average(vals, weights=wts),
            'weighted_median': sv[wmi],
            'p25': np.percentile(vals, 25), 'p75': np.percentile(vals, 75),
            'n_wells': len(vals),
        }
    print(f"\n{'='*95}")
    print(f"AGGREGATE WELL-LEVEL: {split_name.upper()}")
    print(f"{'='*95}")
    print(f"{'Metric':<15} {'Unwt Mean':<12} {'Unwt Med':<12} {'Wt Mean':<12} {'Wt Med':<12} {'IQR':<20}")
    print(f"{'-'*95}")
    for m in ['r2', 'rmse', 'mae', 'nse', 'kge', 'rmsle', 'mape', 'rrmse', 'abs_norm_bias']:
        if m in agg:
            s = agg[m]
            print(f"{m.upper():<15} {s['unweighted_mean']:<12.4f} {s['unweighted_median']:<12.4f} "
                  f"{s['weighted_mean']:<12.4f} {s['weighted_median']:<12.4f} "
                  f"{s['p25']:.3f} - {s['p75']:.3f}")
    print(f"{'='*95}\n")
    return agg


def plot_well_cdfs(split_name, df_split, filename, save_dir):
    metrics_to_plot = [
        ('rmse', f'RMSE ({CONFIG["label_units"]})'), ('mae', f'MAE ({CONFIG["label_units"]})'),
        ('smpe', 'Signed Mean Percent Error (%)'), ('rmsle', 'RMSLE'),
        ('relative_error', 'Relative Error'), ('nse', 'NSE'),
        ('kge', 'KGE'), ('rrmse', 'Relative RMSE'),
        ('abs_norm_bias', 'Absolute Normalized Bias'),
    ]
    x_limits = {
        'rmse': (0, 0.005), 'mae': (0, 0.005), 'smpe': (-200, 200),
        'rmsle': (0, 2), 'relative_error': (0, 3),
        'nse': (-1, 1), 'kge': (-1, 1), 'rrmse': (0, 3), 'abs_norm_bias': (0, 2),
    }
    colors = {"train": "#2E86C1", "test": "#E74C3C"}
    n_cols = 3
    n_rows = (len(metrics_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 6 * n_rows))
    axes = axes.flatten()
    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx]
        vals = df_split[metric].dropna().values
        if len(vals) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        n_samp = df_split.loc[df_split[metric].notna(), 'n_samples'].values
        si = np.argsort(vals)
        sv = vals[si]
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv, cdf, color=colors[split_name], linewidth=3, alpha=0.9,
                label=f"{split_name.capitalize()} ({len(vals)} wells)")
        med = np.median(vals)
        sw = n_samp[si]
        cs = np.cumsum(sw)
        wm = sv[np.searchsorted(cs, cs[-1] / 2.0)]
        ax.legend(loc="upper left", fontsize=10)
        ax.text(0.98, 0.05, f"Median:     {med:.3f}\nWtd Median: {wm:.3f}",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                fontweight='bold', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                          edgecolor=colors[split_name], linewidth=2.5))
        ax.set_xlabel(label, fontweight='bold', fontsize=11)
        ax.set_ylabel("Cumulative Probability", fontweight='bold', fontsize=11)
        ax.set_title(f"CDF: {label} ({split_name.capitalize()})", fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        if metric in x_limits:
            ax.set_xlim(x_limits[metric])
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax.axhline(0.67, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        if metric == 'smpe':
            ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
    for idx in range(len(metrics_to_plot), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"✓ Saved {filename}")
    plt.show()


def plot_combined_key_boxplots(test_well_metrics, filename, save_dir):
    print('TEST WELL METRICS: ', test_well_metrics.describe())
    metrics_to_plot = [
        ('relative_error', 'Relative\nError',   '#7a7a7a'),
        ('abs_norm_bias',  'Abs. Norm.\nBias',   '#7a7a7a'),
        ('mape',           'MAPE',               '#7a7a7a'),
        ('rmsle',          'RMSLE',              '#7a7a7a'),
    ]

    plt.rcParams.update({
        'font.family':        'serif',
        'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset':   'dejavuserif',
        'font.size':          36,
        'axes.labelsize':     48,
        'axes.titlesize':     52,
        'xtick.labelsize':    40,
        'ytick.labelsize':    36,
        'figure.dpi':         150,
        'savefig.dpi':        300,
        'axes.linewidth':     2.8,
        'xtick.major.width':  2.4,
        'ytick.major.width':  2.4,
        'xtick.major.size':   14,
        'ytick.major.size':   14,
        'ytick.minor.size':   8,
    })

    fig, ax = plt.subplots(figsize=(36, 30))

    box_data, box_positions, box_colors, box_labels = [], [], [], []

    for i, (metric, label, color) in enumerate(metrics_to_plot):
        vals = test_well_metrics[metric].dropna().values.copy()
        if len(vals) == 0:
            continue
        if metric == 'smpe':
            vals = vals / 100.0
        box_data.append(vals)
        box_positions.append(i)
        box_colors.append(color)
        box_labels.append(label)

    # Violins
    for i, (vals, color) in enumerate(zip(box_data, box_colors)):
        vals_clipped = np.clip(vals, -2, 2)
        if len(vals_clipped) < 2:
            continue
        vp = ax.violinplot(vals_clipped, positions=[i],
                           showmedians=False, showextrema=False, widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.30)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)

    # Boxplots
    bp = ax.boxplot(
        box_data, positions=box_positions, widths=0.32, patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.3,
                        markeredgewidth=0.5, markeredgecolor='dimgray'),
        medianprops=dict(color='black', linewidth=3.0),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
    )

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
        patch.set_edgecolor('#2a2a2a')
        patch.set_linewidth(2.0)
    for whisker in bp['whiskers']:
        whisker.set_color('#2a2a2a')
        whisker.set_linewidth(1.8)
    for cap in bp['caps']:
        cap.set_color('#2a2a2a')
        cap.set_linewidth(1.8)

    # Annotations — med + IQR on line 1, n on line 2, font 2x larger
    y_lim_top = 2.0
    for i, (vals, color) in enumerate(zip(box_data, box_colors)):
        med = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        n = len(vals)
        whisker_top = min(q75 + 1.5 * (q75 - q25), np.max(vals))
        whisker_top = min(whisker_top, y_lim_top)
        y_text = min(whisker_top + 0.10, y_lim_top - 0.05)
        ax.text(i, y_text,
                f'med = {med:.2f} [{q25:.2f}, {q75:.2f}]\n$n$ = {n}',
                ha='center', va='bottom',
                fontsize=50, fontfamily='serif', linespacing=1.3,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          alpha=0.95, edgecolor='#2a2a2a', linewidth=2.0))

    ax.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=1.4, zorder=0)
    ax.set_xticks(box_positions)
    ax.set_xticklabels(box_labels, fontsize=64)
    ax.set_ylabel('Metric Value', fontsize=72, labelpad=14)
    ax.set_ylim(0, y_lim_top)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))
    ax.tick_params(axis='y', which='both', direction='out')
    ax.tick_params(axis='y', labelsize=64)
    ax.grid(True, axis='y', which='major', alpha=0.3, linestyle='-', linewidth=0.7)
    ax.grid(True, axis='y', which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2.8)
    ax.spines['bottom'].set_linewidth(2.8)
    fig.subplots_adjust(bottom=0.15)

    plt.savefig(os.path.join(save_dir, filename), dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved {filename}")
    plt.show()
    plt.close(fig)


# ============================================================================
# ── UQ: SIMPLE CONFORMAL PREDICTION DIAGNOSTICS
# ============================================================================

def plot_conformal_diagnostics(
    y_true_orig, y_pred_orig, y_true_log, y_pred_log,
    conformal_radii, uq_levels, config, save_dir,
):
    """
    Six-panel diagnostic figure for split conformal prediction.

    Panels:
      (0,0) Scatter + 90% CI bars
      (0,1) Coverage calibration
      (0,2) Residual histogram in log-space + conformal radii
      (1,0) Relative width vs prediction magnitude
      (1,1) CI span in orders of magnitude (CDF)
      (1,2) Missed points analysis — where does the CI fail?
    """

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Use 90% level for the scatter
    r90 = conformal_radii.get(0.90, conformal_radii[max(uq_levels)])
    lower_90 = np.exp(y_pred_log - r90)
    upper_90 = np.exp(y_pred_log + r90)

    # ── (0,0) Scatter + 90% CI bars ──
    ax = axes[0, 0]
    n_show = min(500, len(y_true_orig))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y_true_orig), n_show, replace=False)
    idx = idx[np.argsort(y_true_orig[idx])]
    ax.scatter(y_true_orig[idx], y_pred_orig[idx], s=12, alpha=0.5,
               color='steelblue', zorder=3, label='Predictions')
    ax.vlines(y_true_orig[idx], lower_90[idx], upper_90[idx],
              alpha=0.12, color='steelblue', linewidth=0.8)
    mn = min(y_true_orig.min(), lower_90.min()) * 0.8
    mx = max(y_true_orig.max(), upper_90.max()) * 1.2
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2, zorder=1000)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(f'Actual ({config["label_units"]})')
    ax.set_ylabel(f'Predicted ({config["label_units"]})')
    cov_90 = np.mean((y_true_orig >= lower_90) & (y_true_orig <= upper_90))
    ax.set_title(f'90% Conformal Intervals (coverage={cov_90:.3f})')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    # ── (0,1) Coverage calibration ──
    ax = axes[0, 1]
    target_covs, empirical_covs = [], []
    for level in sorted(uq_levels):
        r = conformal_radii[level]
        lo = np.exp(y_pred_log - r)
        hi = np.exp(y_pred_log + r)
        cov = np.mean((y_true_orig >= lo) & (y_true_orig <= hi))
        target_covs.append(level)
        empirical_covs.append(cov)
    ax.plot([0.5, 1], [0.5, 1], 'k--', lw=1.5, label='Perfect')
    ax.scatter(target_covs, empirical_covs, s=150, c='steelblue',
               edgecolors='navy', zorder=3, label='Conformal')
    for t, e in zip(target_covs, empirical_covs):
        ax.annotate(f'{t:.0%} → {e:.0%}', (t, e),
                    textcoords="offset points", xytext=(10, 5), fontsize=11,
                    fontweight='bold')
    ax.set_xlabel('Target Coverage'); ax.set_ylabel('Empirical Coverage')
    ax.set_title('Coverage Calibration')
    ax.set_xlim(0.55, 1.02); ax.set_ylim(0.55, 1.02)
    ax.legend(); ax.grid(True, alpha=0.3)

    # ── (0,2) Residual histogram + conformal radii ──
    ax = axes[0, 2]
    residuals_log = np.abs(y_true_log - y_pred_log)
    ax.hist(residuals_log, bins=60, density=True, alpha=0.7,
            color='steelblue', edgecolor='navy', label='|residuals| (log-space)')
    colors_uq = ['#27AE60', '#E74C3C', '#8E44AD']
    for i, level in enumerate(sorted(uq_levels)):
        r = conformal_radii[level]
        pct = int(level * 100)
        ax.axvline(r, color=colors_uq[i % 3], ls='--', lw=2.5,
                   label=f'{pct}% radius = {r:.3f}')
    ax.set_xlabel('|Residual| (log-space)')
    ax.set_ylabel('Density')
    ax.set_title('Calibration Residual Distribution')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── (1,0) Relative width vs prediction magnitude ──
    ax = axes[1, 0]
    for i, level in enumerate(sorted(uq_levels)):
        r = conformal_radii[level]
        rel_w = (np.exp(y_pred_log + r) - np.exp(y_pred_log - r)) / y_pred_orig
        pct = int(level * 100)
        ax.scatter(y_pred_orig, rel_w, s=4, alpha=0.15,
                   color=colors_uq[i % 3], label=f'{pct}% (med={np.median(rel_w):.1%})')
    ax.set_xscale('log')
    ax.set_xlabel(f'Predicted CO₂ ({config["label_units"]})')
    ax.set_ylabel('Relative Interval Width')
    ax.set_title('Relative Width vs Prediction')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── (1,1) CI span in orders of magnitude (CDF) ──
    ax = axes[1, 1]
    for i, level in enumerate(sorted(uq_levels)):
        r = conformal_radii[level]
        span_log10 = 2 * r / np.log(10)
        pct = int(level * 100)
        ax.axvline(span_log10, color=colors_uq[i % 3], lw=3,
                   label=f'{pct}%: {span_log10:.2f} orders of magnitude')
    ax.set_xlabel('CI Span (orders of magnitude)')
    ax.set_title('Interval Width in Orders of Magnitude')
    ax.set_xlim(0, 3)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # ── (1,2) Where does the CI miss? ──
    ax = axes[1, 2]
    r95 = conformal_radii.get(0.95, conformal_radii[max(uq_levels)])
    lo_95 = np.exp(y_pred_log - r95)
    hi_95 = np.exp(y_pred_log + r95)
    covered = (y_true_orig >= lo_95) & (y_true_orig <= hi_95)
    missed  = ~covered

    ax.scatter(y_true_orig[covered], y_pred_orig[covered], s=8, alpha=0.15,
               color='steelblue', label=f'Covered ({covered.sum():,})')
    if missed.sum() > 0:
        ax.scatter(y_true_orig[missed], y_pred_orig[missed], s=25, alpha=0.7,
                   color='red', edgecolors='darkred', linewidth=0.5, zorder=5,
                   label=f'Missed ({missed.sum():,})')
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(f'Actual ({config["label_units"]})')
    ax.set_ylabel(f'Predicted ({config["label_units"]})')
    ax.set_title(f'95% CI: Covered vs Missed Points')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = 'conformal_diagnostics.png'
    plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches='tight')
    print(f"✓ Saved {fname}")
    plt.show()


# ============================================================================
# MAIN TRAINING
# ============================================================================

def train(df_cleaned, config=CONFIG):
    """
    Main training entry point.

    Returns
    -------
    pipeline : dict   — everything needed for inference (including uncertainty)
    save_dirs : dict  — paths to saved artifacts
    """

    # --- Setup directories ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['run_name']}_{timestamp}" if config['run_name'] else f"run_{timestamp}"
    base_dir = config['save_directory']
    run_dir  = os.path.join(base_dir, run_name)
    save_dirs = {
        'run':        run_dir,
        'models':     os.path.join(run_dir, 'models'),
        'dataframes': os.path.join(run_dir, 'dataframes'),
        'plots':      os.path.join(run_dir, 'plots'),
        'logs':       os.path.join(run_dir, 'logs'),
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)
    print(f"Run: {run_name}")

    features = list(NUMERIC_FEATURES)

    # --- Temporal features ---
    print("Adding temporal features...")
    df_cleaned = add_temporal_features(df_cleaned, 'Date')

    # --- Interaction terms ---
    print("Computing interaction terms...")
    df_cleaned = create_interaction_terms(df_cleaned)

    y          = df_cleaned[config['label']].copy()
    pfaf_ids   = df_cleaned['PFAF_ID'].copy()
    dates      = df_cleaned['Date'].copy()
    y_original = y.copy()
    y          = np.log(y)

    X = df_cleaned[features].copy()
    available_cat_cols = [c for c in config['categorical_features'] if c in df_cleaned.columns]
    X_categorical = df_cleaned[available_cat_cols].copy() if available_cat_cols else pd.DataFrame()
    well_ids = df_cleaned[config['well_column']].copy() if config['well_column'] in df_cleaned.columns \
               else pd.Series(range(len(df_cleaned)), name='dummy_well_id')

    initial_count = len(X)

    # --- Filter missing target ---
    mask = y.notna()
    X, y, pfaf_ids, well_ids, dates, y_original = [
        a[mask].reset_index(drop=True)
        for a in [X, y, pfaf_ids, well_ids, dates, y_original]
    ]
    if not X_categorical.empty:
        X_categorical = X_categorical[mask].reset_index(drop=True)
    print(f"Initial: {initial_count:,} | After CO2 filter: {len(X):,}")

    # --- CO2 thresholds ---
    if config['threshold_co2'] is not None:
        co2_mask = y_original >= config['threshold_co2']
        X, y, pfaf_ids, well_ids, dates, y_original = [
            a[co2_mask].reset_index(drop=True)
            for a in [X, y, pfaf_ids, well_ids, dates, y_original]
        ]
        if not X_categorical.empty:
            X_categorical = X_categorical[co2_mask].reset_index(drop=True)
        print(f"After CO2 >= {config['threshold_co2']}: {len(X):,}")

    if config['threshold_co2_high'] is not None:
        co2_mask = y_original <= config['threshold_co2_high']
        X, y, pfaf_ids, well_ids, dates, y_original = [
            a[co2_mask].reset_index(drop=True)
            for a in [X, y, pfaf_ids, well_ids, dates, y_original]
        ]
        if not X_categorical.empty:
            X_categorical = X_categorical[co2_mask].reset_index(drop=True)
        print(f"After CO2 <= {config['threshold_co2_high']}: {len(X):,}")

    # --- Location count filter ---
    if config.get('filter_by_location_count', False):
        loc_counts = well_ids.value_counts()
        keep = loc_counts[loc_counts >= config['min_samples_per_location']].index
        loc_mask = well_ids.isin(keep)
        X, y, pfaf_ids, well_ids, dates, y_original = [
            a[loc_mask].reset_index(drop=True)
            for a in [X, y, pfaf_ids, well_ids, dates, y_original]
        ]
        if not X_categorical.empty:
            X_categorical = X_categorical[loc_mask].reset_index(drop=True)
        print(f"Well filter (≥{config['min_samples_per_location']}): {len(X):,}")

    # --- NA removal ---
    missing_per_row = X.isnull().sum(axis=1)
    rows_to_remove = np.zeros(len(X), dtype=bool)
    if config['remove_rows_with_all_na']:
        rows_to_remove |= (missing_per_row == len(features))
    if config['remove_rows_with_most_na']:
        rows_to_remove |= (missing_per_row >= int(len(features) * config['most_na_threshold']))
    if config['na_removal_threshold'] is not None:
        rows_to_remove |= (missing_per_row > int(len(features) * config['na_removal_threshold']))

    X, y, pfaf_ids, well_ids, dates, y_original = [
        a[~rows_to_remove].reset_index(drop=True)
        for a in [X, y, pfaf_ids, well_ids, dates, y_original]
    ]
    if not X_categorical.empty:
        X_categorical = X_categorical[~rows_to_remove].reset_index(drop=True)
    print(f"After NA removal: {len(X):,}")

    # --- Outlier removal ---
    if config['outlier_removal_method'] is not None:
        outlier_matrix = pd.DataFrame(False, index=X.index, columns=features)
        for feat in features:
            fdata = X[feat].dropna()
            if len(fdata) < 10:
                continue
            if config['outlier_removal_method'] == 'iqr':
                q25, q75 = np.percentile(fdata, [25, 75])
                iqr = q75 - q25
                outlier_matrix[feat] = (
                    (X[feat] < q25 - config['outlier_threshold'] * iqr) |
                    (X[feat] > q75 + config['outlier_threshold'] * iqr)
                )
            elif config['outlier_removal_method'] == 'zscore':
                z = np.abs((X[feat] - fdata.mean()) / fdata.std())
                outlier_matrix[feat] = z > config['outlier_threshold']
        outlier_mask = outlier_matrix.sum(axis=1) >= config['min_features_for_outlier']
        X, y, pfaf_ids, well_ids, dates, y_original = [
            a[~outlier_mask].reset_index(drop=True)
            for a in [X, y, pfaf_ids, well_ids, dates, y_original]
        ]
        if not X_categorical.empty:
            X_categorical = X_categorical[~outlier_mask].reset_index(drop=True)
        print(f"Outlier removal: {outlier_mask.sum():,} removed → {len(X):,}")

    # --- Well-grouped train/test split ---
    print("\nWell-grouped train/test split...")
    well_stats = pd.DataFrame({'well_id': well_ids, config['label']: y})
    well_stats = well_stats.groupby('well_id').agg({config['label']: 'median'}).reset_index()
    well_stats.columns = ['well_id', 'median_co2']
    well_stats['co2_bin'] = pd.qcut(well_stats['median_co2'], q=5, labels=False, duplicates='drop')

    train_wells, test_wells = train_test_split(
        well_stats['well_id'], test_size=config['test_size'],
        stratify=well_stats['co2_bin'], random_state=config['random_state'])

    train_mask = well_ids.isin(train_wells)
    test_mask  = well_ids.isin(test_wells)

    X_train_num = X[train_mask].reset_index(drop=True)
    X_test_num  = X[test_mask].reset_index(drop=True)
    X_train_cat = X_categorical[train_mask].reset_index(drop=True) if not X_categorical.empty else pd.DataFrame()
    X_test_cat  = X_categorical[test_mask].reset_index(drop=True)  if not X_categorical.empty else pd.DataFrame()

    y_train     = y[train_mask].reset_index(drop=True)
    y_test      = y[test_mask].reset_index(drop=True)
    pfaf_train  = pfaf_ids[train_mask].reset_index(drop=True)
    pfaf_test   = pfaf_ids[test_mask].reset_index(drop=True)
    well_train  = well_ids[train_mask].reset_index(drop=True)
    well_test   = well_ids[test_mask].reset_index(drop=True)
    dates_train = dates[train_mask].reset_index(drop=True)
    dates_test  = dates[test_mask].reset_index(drop=True)
    y_train_orig = y_original[train_mask].reset_index(drop=True)
    y_test_orig  = y_original[test_mask].reset_index(drop=True)

    assert len(set(well_train) & set(well_test)) == 0, "Data leakage!"
    print(f"Train: {len(X_train_num):,} ({well_train.nunique():,} wells)")
    print(f"Test:  {len(X_test_num):,} ({well_test.nunique():,} wells)")

    # --- Build pipeline dict and fit transforms ---
    pipeline = {
        'config': config,
        'numeric_features': features,
        'interaction_defs': INTERACTION_DEFS,
        'imputers': {},
        'scalers': {},
        'encoders': {},
        'encoded_feature_names': [],
        'all_feature_names': [],
    }

    # Impute
    X_train_imp, pipeline['imputers'] = impute_numeric(X_train_num, {}, features, fit=True)
    X_test_imp, _ = impute_numeric(X_test_num, pipeline['imputers'], features, fit=False)

    # Scale
    X_train_scl, pipeline['scalers'] = scale_numeric(X_train_imp, {}, features, fit=True)
    X_test_scl, _ = scale_numeric(X_test_imp, pipeline['scalers'], features, fit=False)

    # Encode categoricals
    X_train_cat_enc, pipeline['encoders'], pipeline['encoded_feature_names'] = encode_categoricals(
        X_train_cat, {}, config, y_train=y_train, fit=True)
    X_test_cat_enc, _, _ = encode_categoricals(
        X_test_cat, pipeline['encoders'], config, fit=False)

    # Combine
    if X_train_cat_enc.size > 0:
        X_train_final = np.hstack([X_train_scl.values, X_train_cat_enc])
        X_test_final  = np.hstack([X_test_scl.values,  X_test_cat_enc])
    else:
        X_train_final = X_train_scl.values
        X_test_final  = X_test_scl.values
    all_feature_names = features + pipeline['encoded_feature_names']
    pipeline['all_feature_names'] = all_feature_names
    n_cat = X_train_cat_enc.shape[1] if X_train_cat_enc.ndim == 2 else 0
    print(f"Features: {len(all_feature_names)} ({len(features)} numeric + {n_cat} categorical)\n")

    # --- Train/test distribution analysis ---
    if config['analyze_test_distributions']:
        print("="*80 + "\nTRAIN vs TEST DISTRIBUTION\n" + "="*80)
        for lbl, fn in [('Mean', np.mean), ('Median', np.median), ('Std', np.std)]:
            print(f"  {lbl}: train={fn(y_train_orig):.6f}, test={fn(y_test_orig):.6f}")
        ks_stats = []
        for feat in features:
            tv  = X_train_scl[feat].dropna().values
            tsv = X_test_scl[feat].dropna().values
            if len(tv) > 0 and len(tsv) > 0:
                ks, p = ks_2samp(tv, tsv)
                ks_stats.append({'feature': feat, 'ks_statistic': ks, 'p_value': p})
        ks_df = pd.DataFrame(ks_stats).sort_values('ks_statistic', ascending=False)
        print(f"\nTop 10 KS differences:")
        for _, row in ks_df.head(10).iterrows():
            print(f"  {row['feature']:<25} KS={row['ks_statistic']:.4f}  p={row['p_value']:.2e}")

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        for i, feat in enumerate(ks_df.head(16)['feature']):
            axes[i].hist(X_train_scl[feat].values, bins=50, alpha=0.5, label='Train', density=True, color='blue')
            axes[i].hist(X_test_scl[feat].values,  bins=50, alpha=0.5, label='Test',  density=True, color='red')
            axes[i].set_title(f'{feat}\n(KS={ks_df[ks_df["feature"]==feat]["ks_statistic"].values[0]:.3f})', fontsize=9)
            axes[i].legend(fontsize=7)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dirs['plots'], 'train_test_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # --- Model training ---
    def get_xgb_model():
        return XGBRegressor(**config['xgb_params'])

    print("Training XGBoost...")
    model = get_xgb_model()
    model.fit(X_train_final, y_train)
    pipeline['model'] = model
    print(f"✓ Trained (max_depth={config['xgb_params']['max_depth']})")

    # ============================================================
    # --- Cross-validation ---
    # Captures: per-fold sample-level metrics (R², RMSE, MAE),
    #           per-fold feature importances (for error bars),
    #           per-fold well-level median R² (for stability check)
    # ============================================================
    cv_results_dict = {}
    fold_importances = None  # will be (n_folds, n_features) if CV runs
    well_r2_per_fold = []
    if config['perform_cv']:
        gkfold = GroupKFold(n_splits=config['cv_folds'])
        print(f"\n{config['cv_folds']}-fold grouped CV...")
        r2s, rmses, maes = [], [], []
        kges, rmsles, smpes, rel_errors, mapes = [], [], [], [], []
        fold_importances_list = []
        for fold, (ti, vi) in enumerate(gkfold.split(X_train_final, y_train, groups=well_train), 1):
            fm = get_xgb_model()
            fm.fit(X_train_final[ti], y_train.iloc[ti])
            yp = np.exp(fm.predict(X_train_final[vi]))
            yv = np.exp(y_train.iloc[vi].values)
            fold_m = calculate_comprehensive_metrics(yv, yp)
            r2s.append(fold_m['r2'])
            rmses.append(fold_m['rmse'])
            maes.append(fold_m['mae'])
            kges.append(fold_m['kge'])
            rmsles.append(fold_m['rmsle'])
            smpes.append(fold_m['smpe'])
            rel_errors.append(fold_m['relative_error'])
            mapes.append(fold_m['mape'])
            fold_importances_list.append(fm.feature_importances_)

            # Well-level stability
            fold_results = pd.DataFrame({
                'MonitoringLocation': well_train.iloc[vi].values,
                'CO2_actual': yv,
                'CO2_predicted': yp,
            })
            fold_wm = calculate_well_level_metrics(
                fold_results, f'cv_fold_{fold}', min_samples=10)
            if len(fold_wm) > 0 and fold_wm['r2'].notna().any():
                well_r2_per_fold.append(np.median(fold_wm['r2'].dropna()))
            else:
                well_r2_per_fold.append(np.nan)

            print(f"  Fold {fold}: R²={r2s[-1]:.4f} RMSE={rmses[-1]:.2e} "
                  f"well_med_R²={well_r2_per_fold[-1]:.4f}")

        r2s, rmses, maes = np.array(r2s), np.array(rmses), np.array(maes)
        kges, rmsles, smpes, rel_errors, mapes = (
            np.array(kges), np.array(rmsles), np.array(smpes), np.array(rel_errors), np.array(mapes))
        fold_importances = np.array(fold_importances_list)
        well_r2_per_fold = np.array(well_r2_per_fold)
        wr2_clean = well_r2_per_fold[~np.isnan(well_r2_per_fold)]

        print(f"\n  CV sample-level R²:    {r2s.mean():.4f} ± {r2s.std()*2:.4f} (95% range)")
        if len(wr2_clean) > 0:
            print(f"  CV well-level med R²:  {wr2_clean.mean():.4f} ± {wr2_clean.std()*2:.4f} (95% range)")
        cv_results_dict = {
            'r2_scores': r2s, 'rmse_scores': rmses, 'mae_scores': maes,
            'kge_scores': kges, 'rmsle_scores': rmsles,
            'smpe_scores': smpes, 'rel_error_scores': rel_errors,
            'fold_importances': fold_importances,
            'well_r2_per_fold': well_r2_per_fold,
        }
        pipeline['cv_fold_importances'] = fold_importances
        pipeline['cv_well_r2_per_fold'] = well_r2_per_fold

    # --- Evaluate ---
    train_pred_orig = np.exp(model.predict(X_train_final))
    test_pred_orig  = np.exp(model.predict(X_test_final))

    train_metrics = calculate_comprehensive_metrics(y_train_orig.values, train_pred_orig)
    test_metrics  = calculate_comprehensive_metrics(y_test_orig.values,  test_pred_orig)
    print_comprehensive_metrics(train_metrics, "Train")
    print_comprehensive_metrics(test_metrics,  "Test")

    # --- Save predictions ---
    train_results_df = pd.DataFrame({
        'Date': dates_train.values, 'MonitoringLocation': well_train.values,
        'PFAF_ID': pfaf_train.values,
        'CO2_actual': y_train_orig.values, 'CO2_predicted': train_pred_orig, 'split': 'train'
    })
    test_results_df = pd.DataFrame({
        'Date': dates_test.values, 'MonitoringLocation': well_test.values,
        'PFAF_ID': pfaf_test.values,
        'CO2_actual': y_test_orig.values, 'CO2_predicted': test_pred_orig, 'split': 'test'
    })

    # --- Well-level metrics ---
    MIN_SAMP = 10
    print(f"\nWell-level metrics (n≥{MIN_SAMP})...")
    train_wm = calculate_well_level_metrics(train_results_df, "train", min_samples=MIN_SAMP)
    test_wm  = calculate_well_level_metrics(test_results_df,  "test",  min_samples=MIN_SAMP)
    all_wm   = pd.concat([train_wm, test_wm], ignore_index=True)
    compute_aggregate_well_stats(train_wm, "train")
    compute_aggregate_well_stats(test_wm,  "test")
    all_wm.to_csv(os.path.join(save_dirs['dataframes'], 'well_level_metrics.csv'), index=False)

    if len(train_wm) > 0:
        plot_well_cdfs("train", train_wm, "well_level_cdfs_train.png", save_dirs['plots'])
    if len(test_wm) > 0:
        plot_well_cdfs("test",  test_wm,  "well_level_cdfs_test.png",  save_dirs['plots'])
    plot_combined_key_boxplots(test_wm, 'combined_key_cdfs.png', save_dirs['plots'])

    # --- Feature importance (grouped) ---
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    onehot_prefix_map = {}
    onehot_cols = [c for c in available_cat_cols if config['encoding_strategy'].get(c) == 'onehot']
    if 'onehot' in pipeline['encoders']:
        for fname in pipeline['encoders']['onehot'].get_feature_names_out(onehot_cols):
            for col in onehot_cols:
                if fname.startswith(col + '_') or fname == col:
                    onehot_prefix_map[fname] = col
                    break

    importance_df['GroupedFeature'] = importance_df['Feature'].map(
        lambda f: onehot_prefix_map.get(f, f))
    importance_grouped = (
        importance_df.groupby('GroupedFeature', as_index=False)['Importance']
        .sum().sort_values('Importance', ascending=False)
        .rename(columns={'GroupedFeature': 'Feature'})
    )

    # --- CV-based fold importance stats (mean and std per raw feature) ---
    # cv_imp_lookup: raw_feature_name → (mean, std) across folds
    cv_imp_lookup = {}
    if fold_importances is not None:
        imp_mean = fold_importances.mean(axis=0)
        imp_std  = fold_importances.std(axis=0)
        cv_imp_lookup = dict(zip(all_feature_names, zip(imp_mean, imp_std)))

    def _grouped_imp_stats(grouped_name):
        """Return (mean, std) for a grouped feature, summing means and
        combining stds in quadrature across constituent raw features."""
        if not cv_imp_lookup:
            return np.nan, np.nan
        raw = [f for f in all_feature_names
               if onehot_prefix_map.get(f, f) == grouped_name]
        if not raw:
            return np.nan, np.nan
        means = [cv_imp_lookup[f][0] for f in raw]
        stds  = [cv_imp_lookup[f][1] for f in raw]
        return float(np.sum(means)), float(np.sqrt(np.sum(np.square(stds))))

    ordinal_cols = [c for c in available_cat_cols if config['encoding_strategy'].get(c) == 'ordinal']
    target_cols  = [c for c in available_cat_cols if config['encoding_strategy'].get(c) == 'target']
    temporal_feats = {'month_sin', 'month_cos', 'day_sin', 'day_cos', 'season', 'year'}
    cat_feats      = set(onehot_cols + ordinal_cols + target_cols)

    print("\nTop 20 Features (grouped):")
    for i, (_, row) in enumerate(importance_grouped.head(20).iterrows(), 1):
        tag = "TEMP: " if row['Feature'] in temporal_feats else \
              "CAT:  " if row['Feature'] in cat_feats else "      "
        m, s = _grouped_imp_stats(row['Feature'])
        if not np.isnan(s):
            print(f"  {i:>2}. {tag}{row['Feature']:<30} {row['Importance']:.4f}  "
                  f"(CV: {m:.4f} ± {s:.4f})")
        else:
            print(f"  {i:>2}. {tag}{row['Feature']:<30} {row['Importance']:.4f}")

    # --- RFECV ---
    print(f"\n{'='*80}\nRFECV — numeric features only\n{'='*80}")
    rfe_estimator = XGBRegressor(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=3.0,
        random_state=config['random_state'], n_jobs=-1, tree_method='hist')

    rfecv = RFECV(
        estimator=rfe_estimator, step=3,
        cv=GroupKFold(n_splits=config['cv_folds']),
        scoring='r2', min_features_to_select=10, n_jobs=-1, verbose=1)

    X_train_num_only = X_train_scl.values
    X_test_num_only  = X_test_scl.values
    rfecv.fit(X_train_num_only, y_train, groups=well_train)

    selected_mask = rfecv.support_
    selected_features = [features[i] for i, s in enumerate(selected_mask) if s]
    dropped_features  = [features[i] for i, s in enumerate(selected_mask) if not s]
    print(f"\n Optimal: {rfecv.n_features_}/{len(features)}")
    print(f"Selected: {selected_features}")
    print(f"Dropped: {dropped_features}")

    # RFECV plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cv_mean = rfecv.cv_results_['mean_test_score']
    cv_std  = rfecv.cv_results_['std_test_score']
    x_vals  = list(range(rfecv.min_features_to_select,
                         rfecv.min_features_to_select + len(cv_mean) * rfecv.step, rfecv.step))
    ax.plot(x_vals, cv_mean, 'o-', color='steelblue', linewidth=2, markersize=5)
    ax.fill_between(x_vals, cv_mean - cv_std, cv_mean + cv_std, alpha=0.2, color='steelblue')
    ax.axvline(rfecv.n_features_, color='red', linestyle='--', linewidth=2,
               label=f'Optimal: {rfecv.n_features_}')
    ax.set_xlabel('N Features'); ax.set_ylabel('CV R²'); ax.set_title('RFECV')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dirs['plots'], 'rfecv_scores.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Retrain on reduced
    if X_train_cat_enc.size > 0:
        X_train_rfe = np.hstack([X_train_num_only[:, selected_mask], X_train_cat_enc])
        X_test_rfe  = np.hstack([X_test_num_only[:,  selected_mask], X_test_cat_enc])
    else:
        X_train_rfe = X_train_num_only[:, selected_mask]
        X_test_rfe  = X_test_num_only[:,  selected_mask]
    rfe_feature_names = selected_features + pipeline['encoded_feature_names']

    model_rfe = get_xgb_model()
    model_rfe.fit(X_train_rfe, y_train)
    rfe_train_pred = np.exp(model_rfe.predict(X_train_rfe))
    rfe_test_pred  = np.exp(model_rfe.predict(X_test_rfe))
    rfe_train_metrics = calculate_comprehensive_metrics(y_train_orig.values, rfe_train_pred)
    rfe_test_metrics  = calculate_comprehensive_metrics(y_test_orig.values,  rfe_test_pred)

    for lbl, k in [('R²','r2'), ('RMSE','rmse'), ('KGE','kge'), ('RMSLE','rmsle'), ('NSE','nse')]:
        fmt = '.2e' if k == 'rmse' else '.4f'
        print(f"{lbl:<12} {train_metrics[k]:<12{fmt}} {test_metrics[k]:<12{fmt}} "
              f"{rfe_train_metrics[k]:<12{fmt}} {rfe_test_metrics[k]:<12{fmt}}")

    pipeline['model_rfe']             = model_rfe
    pipeline['rfecv']                 = rfecv
    pipeline['rfe_selected_mask']     = selected_mask
    pipeline['rfe_selected_features'] = selected_features
    pipeline['rfe_feature_names']     = rfe_feature_names

    # ================================================================
    # ── UQ BLOCK: SPLIT CONFORMAL PREDICTION
    # ================================================================
    print(f"\n{'='*80}")
    print("UNCERTAINTY QUANTIFICATION — Split Conformal Prediction")
    print(f"{'='*80}")

    uq_levels    = config.get('conformal_levels', [0.68, 0.90, 0.95])
    uq_calib_frac = config.get('conformal_calib_frac', 0.25)

    # ── Calibration split from training wells ──
    train_well_stats_uq = well_stats[well_stats['well_id'].isin(train_wells)].copy()
    try:
        fit_wells_uq, calib_wells_uq = train_test_split(
            train_well_stats_uq['well_id'],
            test_size=uq_calib_frac,
            stratify=train_well_stats_uq['co2_bin'],
            random_state=config['random_state'] + 1)
    except ValueError:
        fit_wells_uq, calib_wells_uq = train_test_split(
            train_well_stats_uq['well_id'],
            test_size=uq_calib_frac,
            random_state=config['random_state'] + 1)

    calib_mask_uq = well_train.isin(calib_wells_uq).values
    X_calib = X_train_final[calib_mask_uq]
    y_calib_log = y_train[calib_mask_uq].values

    n_calib = len(y_calib_log)
    print(f"\n  Calibration set: {n_calib:,} points ({len(calib_wells_uq)} wells)")
    print(f"  Coverage resolution: ±{1/n_calib:.4f}")

    # ── Compute calibration residuals (log-space) ──
    fit_mask_uq = well_train.isin(fit_wells_uq).values
    model_fit_only = get_xgb_model()
    model_fit_only.fit(X_train_final[fit_mask_uq], y_train[fit_mask_uq].values)

    y_pred_calib_log = model_fit_only.predict(X_calib)
    calib_residuals = np.abs(y_calib_log - y_pred_calib_log)

    print(f"\n  Calibration residual stats (log-space):")
    print(f"    Mean:   {np.mean(calib_residuals):.4f}")
    print(f"    Median: {np.median(calib_residuals):.4f}")
    print(f"    Std:    {np.std(calib_residuals):.4f}")
    print(f"    90th:   {np.percentile(calib_residuals, 90):.4f}")
    print(f"    95th:   {np.percentile(calib_residuals, 95):.4f}")

    conformal_radii = {}
    print(f"\n  Conformal radii:")
    for level in sorted(uq_levels):
        q_conf = min(np.ceil((n_calib + 1) * level) / n_calib, 1.0)
        radius = float(np.quantile(calib_residuals, q_conf))
        conformal_radii[level] = radius
        pct = int(level * 100)

        mult_up   = np.exp(radius)
        mult_down = np.exp(-radius)
        span_oom  = 2 * radius / np.log(10)

        print(f"    {pct}%: radius = {radius:.4f} (log-space)")
        print(f"         → prediction × {mult_up:.2f} / ÷ {1/mult_down:.2f}")
        print(f"         → spans {span_oom:.2f} orders of magnitude")

    radii_sorted = [conformal_radii[l] for l in sorted(uq_levels)]
    assert all(radii_sorted[i] <= radii_sorted[i+1]
               for i in range(len(radii_sorted)-1)), \
        f"Nesting violated! radii={radii_sorted}"
    print(f"  ✓ Nesting verified (guaranteed by construction)")

    y_pred_test_log = model.predict(X_test_final)

    print(f"\n  Test set validation:")
    test_ci = {}
    for level in sorted(uq_levels):
        radius = conformal_radii[level]
        lower_orig = np.exp(y_pred_test_log - radius)
        upper_orig = np.exp(y_pred_test_log + radius)
        pct = int(level * 100)
        test_ci[f'CI_lower_{pct}'] = lower_orig
        test_ci[f'CI_upper_{pct}'] = upper_orig

        cov = np.mean((y_test_orig.values >= lower_orig) &
                       (y_test_orig.values <= upper_orig))
        med_width = np.median(upper_orig - lower_orig)
        med_rel_w = np.median((upper_orig - lower_orig) / test_pred_orig)
        print(f"    {pct}%: coverage={cov:.3f} (target={level:.2f}), "
              f"med width={med_width:.2e} {config['label_units']}, "
              f"med rel width={med_rel_w:.1%}")

    # ── Also compute for RFE model ──
    y_pred_calib_rfe_log = model_rfe.predict(X_train_rfe[calib_mask_uq])
    calib_residuals_rfe = np.abs(y_calib_log - y_pred_calib_rfe_log)

    conformal_radii_rfe = {}
    print(f"\n  RFE model conformal radii:")
    for level in sorted(uq_levels):
        q_conf = min(np.ceil((n_calib + 1) * level) / n_calib, 1.0)
        radius = float(np.quantile(calib_residuals_rfe, q_conf))
        conformal_radii_rfe[level] = radius
        pct = int(level * 100)
        print(f"    {pct}%: radius = {radius:.4f}")

    y_pred_test_rfe_log = model_rfe.predict(X_test_rfe)
    print(f"\n  RFE test validation:")
    for level in sorted(uq_levels):
        radius = conformal_radii_rfe[level]
        lo = np.exp(y_pred_test_rfe_log - radius)
        hi = np.exp(y_pred_test_rfe_log + radius)
        cov = np.mean((y_test_orig.values >= lo) & (y_test_orig.values <= hi))
        pct = int(level * 100)
        print(f"    {pct}%: coverage={cov:.3f} (target={level:.2f})")

    pipeline['conformal_radii']     = conformal_radii
    pipeline['conformal_levels']    = uq_levels
    pipeline['rfe_conformal_radii'] = conformal_radii_rfe
    pipeline['n_calib']             = n_calib
    pipeline['uncertainty_method']  = 'split_conformal'

    for col, vals in test_ci.items():
        test_results_df[col] = vals

    all_results_df = pd.concat([train_results_df, test_results_df], ignore_index=True)
    all_results_df.to_csv(os.path.join(save_dirs['dataframes'],
                                        'predictions_with_dates.csv'), index=False)

    plot_conformal_diagnostics(
        y_test_orig.values, test_pred_orig,
        y_test.values, y_pred_test_log,
        conformal_radii, uq_levels, config, save_dirs['plots'])

    # ================================================================
    # END UNCERTAINTY BLOCK
    # ================================================================

    # --- Final figure: three 1:1 plot styles + importance ---
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    purples_scatter = LinearSegmentedColormap.from_list('purples_scatter', [
        '#f3eef6', '#d4c4e0', '#b091c9', '#8b5fb0', '#6a3d9a', '#4a1a7a', '#2d0a4e'])
    purples_density = LinearSegmentedColormap.from_list('purples_topo', [
        '#faf5ff', '#e8d5f5', '#c9a0e0', '#a56bc7', '#7b3fa0', '#551a80', '#2d0050'])
    purples_hex = LinearSegmentedColormap.from_list('purples_hex', [
        '#f7f0fa', '#dbc4eb', '#b88fd4', '#9060b8', '#6b3a96', '#4a1a75', '#280a50'])

    y_test_log_10   = np.log10(np.clip(y_test_orig.values, 1e-10, None))
    test_pred_log10 = np.log10(np.clip(test_pred_orig, 1e-10, None))
    r90 = conformal_radii.get(0.90, conformal_radii[max(uq_levels)])

    if config['perform_cv']:
        stats_text = (
            f"Rel.E = {rel_errors.mean():.4f} ± {rel_errors.std():.2f}\n"
            f"R²    = {r2s.mean():.2f} ± {r2s.std():.2f}\n"
            f"MAPE  = {mapes.mean():.2f} ± {mapes.std():.2f}%\n"
            f"RMSLE = {rmsles.mean():.2f} ± {rmsles.std():.2f}\n"
            f"n     = {len(test_pred_orig):,}\n"
        )
    else:
        stats_text = (
            f"Rel.E = {test_metrics['relative_error']:.2f}\n"
            f"R²    = {test_metrics['r2']:.2f}\n"
            f"MAPE  = {mapes.mean():.2f}\n"
            f"RMSLE = {test_metrics['rmsle']:.2f}\n"
            f"Bias  = {test_metrics['mean_bias']:.2e} {config['label_units']}\n"
            f"n     = {len(test_pred_orig):,}\n"
        )

    def _add_stats_box(ax):
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                fontsize=8, fontfamily='monospace', va='bottom', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    fig = plt.figure(figsize=(46, 44))

    # ── (A) KDE density scatter — purple ──
    ax_a = fig.add_subplot(2, 2, 1)
    ax_a.set_facecolor('white')
    xy = np.vstack([y_test_log_10, test_pred_log10])
    kde_vals = gaussian_kde(xy, bw_method=0.15)(xy)
    z_n = (kde_vals - kde_vals.min()) / (kde_vals.max() - kde_vals.min())
    idx_s = z_n.argsort()
    ax_a.scatter(10**y_test_log_10[idx_s], 10**test_pred_log10[idx_s],
                 c=z_n[idx_s], cmap=purples_scatter, s=20, alpha=0.5,
                 edgecolors='none', rasterized=True)
    mn_p = min(y_test_orig.min(), test_pred_orig.min()) * 0.8
    mx_p = max(y_test_orig.max(), test_pred_orig.max()) * 1.2
    ax_a.plot([mn_p, mx_p], [mn_p, mx_p], '--', color='#c0392b', lw=2, label='1:1 line', zorder=1000)
    ax_a.set_xscale('log'); ax_a.set_yscale('log')
    ax_a.set_xlabel(f'Actual ({config["label_units"]})', fontsize=20, fontfamily='serif')
    ax_a.set_ylabel(f'Predicted ({config["label_units"]})', fontsize=20, fontfamily='serif')
    # ax_a.set_title('A.  KDE Density Scatter', fontsize=12, fontfamily='serif',
    #                 fontweight='bold', color='#2d0a4e')
    ax_a.grid(True, alpha=0.3); ax_a.legend(loc='upper left', fontsize=16)
    #_add_stats_box(ax_a)

    # ── (B) Density field + contour lines ──
    ax_b = fig.add_subplot(2, 2, 2)
    ax_b.set_facecolor('#fafafa')
    xedges = np.linspace(y_test_log_10.min() - 0.3, y_test_log_10.max() + 0.3, 120)
    yedges = np.linspace(test_pred_log10.min() - 0.3, test_pred_log10.max() + 0.3, 120)
    H, xe, ye = np.histogram2d(y_test_log_10, test_pred_log10, bins=[xedges, yedges])
    H = gaussian_filter(H.T, sigma=1.8)
    ax_b.imshow(H, origin='lower', aspect='auto', cmap=purples_density,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                interpolation='bilinear')
    contour_levels = np.linspace(H.max() * 0.05, H.max() * 0.9, 8)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ax_b.contour(xc, yc, H, levels=contour_levels,
                 colors='#2d0a4e', linewidths=0.6, alpha=0.6)
    diag_lim = [min(xedges[0], yedges[0]), max(xedges[-1], yedges[-1])]
    ax_b.plot(diag_lim, diag_lim, '--', color='#c0392b', lw=2, zorder=1000)
    ax_b.set_xlim(xedges[0], xedges[-1]); ax_b.set_ylim(yedges[0], yedges[-1])
    ax_b.set_aspect('equal')
    for ao in [ax_b.xaxis, ax_b.yaxis]:
        ao.set_major_locator(mticker.MultipleLocator(0.5))
        ao.set_major_formatter(mticker.FuncFormatter(
            lambda x, p: f'$10^{{{int(x)}}}$' if x == int(x) else ''))
    ax_b.set_xlabel(f'Actual (log₁₀ {config["label_units"]})', fontsize=10, fontfamily='serif')
    ax_b.set_ylabel(f'Predicted (log₁₀ {config["label_units"]})', fontsize=10, fontfamily='serif')
    ax_b.set_title('B.  Density Field + Contours', fontsize=12, fontfamily='serif',
                    fontweight='bold', color='#2d0a4e')
    ax_b.grid(True, alpha=0.15, linewidth=0.4)
    for spine in ax_b.spines.values():
        spine.set_color('#999999'); spine.set_linewidth(0.6)
    _add_stats_box(ax_b)

    # ── (C) Hexagonal binning ──
    ax_c = fig.add_subplot(2, 2, 3)
    ax_c.set_facecolor('#f8f5fa')
    hb = ax_c.hexbin(y_test_log_10, test_pred_log10, gridsize=55, cmap=purples_hex,
                      mincnt=1, linewidths=0.1, edgecolors='#e0d0ea')
    cb = fig.colorbar(hb, ax=ax_c, shrink=0.75, pad=0.02)
    cb.set_label('Count', fontsize=9, fontfamily='serif')
    cb.ax.tick_params(labelsize=8)
    ax_c.plot(diag_lim, diag_lim, '--', color='#c0392b', lw=2, zorder=1000)
    ax_c.set_xlim(xedges[0], xedges[-1]); ax_c.set_ylim(yedges[0], yedges[-1])
    ax_c.set_aspect('equal')
    for ao in [ax_c.xaxis, ax_c.yaxis]:
        ao.set_major_locator(mticker.MultipleLocator(0.5))
        ao.set_major_formatter(mticker.FuncFormatter(
            lambda x, p: f'$10^{{{int(x)}}}$' if x == int(x) else ''))
    ax_c.set_xlabel(f'Actual (log₁₀ {config["label_units"]})', fontsize=10, fontfamily='serif')
    ax_c.set_ylabel(f'Predicted (log₁₀ {config["label_units"]})', fontsize=10, fontfamily='serif')
    ax_c.set_title('C.  Hexagonal Binning', fontsize=12, fontfamily='serif',
                    fontweight='bold', color='#2d0a4e')
    ax_c.grid(True, alpha=0.2, color='#ccbbdd', linewidth=0.4)
    for spine in ax_c.spines.values():
        spine.set_color('#999999'); spine.set_linewidth(0.6)
    _add_stats_box(ax_c)

    # ── (D) Feature importance with CV error bars ──
    ax_d = fig.add_subplot(2, 2, 4)
    top_n = 12
    
    # Compute CV mean/std once on the grouped DataFrame
    importance_grouped['imp_mean'] = importance_grouped['Feature'].apply(
        lambda f: _grouped_imp_stats(f)[0])
    importance_grouped['imp_std'] = importance_grouped['Feature'].apply(
        lambda f: _grouped_imp_stats(f)[1])
    
    # Select top features sorted by CV mean importance (highest first)
    top_features = (
        importance_grouped
        .sort_values('imp_mean', ascending=False)
        .head(top_n)
        .copy()
    )
    
    # Apply display name mapping
    top_features['DisplayName'] = top_features['Feature'].map(
        lambda f: FEATURE_DISPLAY_NAMES.get(f, f))
    
    # --- REMOVED: the duplicate cv_stats concat and redundant re-sort ---
    
    colors_bar = ['orange'       if f in temporal_feats else
                  'mediumorchid' if f in cat_feats else
                  'steelblue'    for f in top_features['Feature']]
    
    if cv_imp_lookup:
        bar_heights = top_features['imp_mean'].values
        bar_errors  = top_features['imp_std'].values
        xlabel_txt  = 'Importance'
    else:
        bar_heights = top_features['Importance'].values
        bar_errors  = None
        xlabel_txt  = 'Importance'
    
    ax_d.barh(range(len(top_features)), bar_heights,
              xerr=bar_errors,
              alpha=0.85, color=colors_bar,
              edgecolor='#2a2a2a', linewidth=1.2,
              error_kw=dict(ecolor='#2a2a2a', lw=1.5, capsize=4, alpha=0.85))
    ax_d.set_yticks(range(len(top_features)))
    ax_d.set_yticklabels(top_features['DisplayName'].values,
                          fontsize=32, fontfamily='serif')
    ax_d.tick_params(axis='x', labelsize=32)
    ax_d.set_xlabel(xlabel_txt, fontsize=36, fontfamily='serif')
    ax_d.grid(True, alpha=0.3, axis='x')
    ax_d.invert_yaxis()  # highest importance at top
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dirs['plots'], 'test_act_pred.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved test_predictions_combined.png")
    plt.show()
    
    # --- Histogram: actual vs predicted CO2 distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    bins_log = np.linspace(
        min(np.log10(y_test_orig.values.min()), np.log10(test_pred_orig.min())) - 0.2,
        max(np.log10(y_test_orig.values.max()), np.log10(test_pred_orig.max())) + 0.2,
        60)
    ax.hist(np.log10(y_test_orig.values), bins=bins_log, alpha=0.5,
            color='steelblue', edgecolor='navy', linewidth=0.5,
            density=True, label=f'Actual (n={len(y_test_orig):,})')
    ax.hist(np.log10(test_pred_orig), bins=bins_log, alpha=0.5,
            color='coral', edgecolor='darkred', linewidth=0.5,
            density=True, label=f'Predicted (n={len(test_pred_orig):,})')
    ax.set_xlabel(f'log₁₀ CO₂ ({config["label_units"]})', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Test Set: Actual vs Predicted (log-space)', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    log_min = np.floor(np.log10(min(y_test_orig.values.min(), test_pred_orig.min())))
    log_max = np.ceil(np.log10(max(y_test_orig.values.max(), test_pred_orig.max())))
    bins_orig = np.logspace(log_min, log_max, 60)
    ax.hist(y_test_orig.values, bins=bins_orig, alpha=0.5,
            color='steelblue', edgecolor='navy', linewidth=0.5,
            density=True, label='Actual')
    ax.hist(test_pred_orig, bins=bins_orig, alpha=0.5,
            color='coral', edgecolor='darkred', linewidth=0.5,
            density=True, label='Predicted')
    ax.set_xscale('log')
    ax.set_xlabel(f'CO₂ ({config["label_units"]})', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Test Set: Actual vs Predicted (original-space)', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dirs['plots'], 'actual_vs_predicted_histograms.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved actual_vs_predicted_featImp.png")
    plt.show()
    
    # --- Final figure: density scatter + importance ---
    from matplotlib.colors import LinearSegmentedColormap

    purples_scatter = LinearSegmentedColormap.from_list('purples_scatter', [
        '#f3eef6', '#d4c4e0', '#b091c9', '#8b5fb0', '#6a3d9a', '#4a1a7a', '#2d0a4e'])

    fig = plt.figure(figsize=(20, 7))

    # ── Left panel: KDE density scatter ──
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('white')
    y_test_log    = np.log10(np.clip(y_test_orig.values, 1e-10, None))
    test_pred_log = np.log10(np.clip(test_pred_orig, 1e-10, None))
    xy = np.vstack([y_test_log, test_pred_log])
    kde_vals = gaussian_kde(xy, bw_method=0.15)(xy)
    z_n = (kde_vals - kde_vals.min()) / (kde_vals.max() - kde_vals.min())
    idx_sort = z_n.argsort()

    scatter = ax1.scatter(10**y_test_log[idx_sort], 10**test_pred_log[idx_sort],
                          c=z_n[idx_sort], s=18, cmap=purples_scatter, alpha=0.55,
                          edgecolors='none', rasterized=True)
    plt.colorbar(scatter, ax=ax1, pad=0.02, shrink=0.85).set_label(
        'Density', fontweight='bold', fontsize=20)

    mn = min(y_test_orig.min(), test_pred_orig.min()) * 0.8
    mx = max(y_test_orig.max(), test_pred_orig.max()) * 1.2
    ax1.plot([mn, mx], [mn, mx], '--', color='#c0392b', lw=2.5, label='1:1 line', zorder=1000)
    ax1.set_xlabel(f'Actual ({config["label_units"]})', fontsize=20, fontfamily='serif')
    ax1.set_ylabel(f'Predicted ({config["label_units"]})', fontsize=20, fontfamily='serif')
    #ax1.set_title('Test: Predictions vs Actual', fontsize=12, fontfamily='serif', fontweight='bold')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3); ax1.legend(loc='upper left', fontsize=10)

    r90 = conformal_radii.get(0.90, conformal_radii[max(uq_levels)])
    if config['perform_cv']:
        stats_text = (
            f"Rel.E = {rel_errors.mean():.2f} ± {rel_errors.std():.2f}\n"
            f"R²    = {r2s.mean():.2f} ± {r2s.std():.2f}\n"
            f"RMSLE = {rmsles.mean():.2f} ± {rmsles.std():.2f}\n"
            f"MAPE  = {mapes.mean():.2f} ± {mapes.std():.2f}%\n"
            f"n     = {len(test_pred_orig):,}\n"
        )
    else:
        stats_text = (
            f"Rel.E = {test_metrics['relative_error']:.2f}\n"
            f"R²    = {test_metrics['r2']:.2f}\n"
            f"RMSLE = {test_metrics['rmsle']:.2f}\n"
            f"MAPE  = {test_metrics['mape']:.2f}%\n"
            f"Bias  = {test_metrics['mean_bias']:.2e} {config['label_units']}\n"
            f"n     = {len(test_pred_orig):,}\n"
        )
    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
             fontsize=14, fontfamily='monospace', va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    # ── Right panel: Feature importance with CV error bars ──
    ax2 = plt.subplot(1, 2, 2)
    top_n = 15

    importance_grouped['imp_mean'] = importance_grouped['Feature'].apply(
        lambda f: _grouped_imp_stats(f)[0])
    importance_grouped['imp_std'] = importance_grouped['Feature'].apply(
        lambda f: _grouped_imp_stats(f)[1])

    top_features = (
        importance_grouped
        .sort_values('imp_mean', ascending=False)
        .head(top_n)
        .copy()
    )
    top_features['DisplayName'] = top_features['Feature'].map(
        lambda f: FEATURE_DISPLAY_NAMES.get(f, f))

    colors_bar = ['orange'       if f in temporal_feats else
                  'mediumorchid' if f in cat_feats else
                  'steelblue'    for f in top_features['Feature']]

    if cv_imp_lookup:
        bar_heights = top_features['imp_mean'].values
        bar_errors  = top_features['imp_std'].values
    else:
        bar_heights = top_features['Importance'].values
        bar_errors  = None

    ax2.barh(range(len(top_features)), bar_heights,
             xerr=bar_errors,
             alpha=0.85, color=colors_bar,
             edgecolor='#2a2a2a', linewidth=1.0,
             error_kw=dict(ecolor='#2a2a2a', lw=1.5, capsize=4, alpha=0.85))
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['DisplayName'].values, fontsize=9, fontfamily='serif')
    ax2.set_xlabel('Importance (CV mean ± std)', fontsize=11, fontfamily='serif')
    ax2.set_title(f'Top {top_n} Features', fontsize=12, fontfamily='serif', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dirs['plots'], 'model_results.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved model_results.png")
    plt.show()
    
    # --- Save everything ---
    metrics = {'train': train_metrics, 'test': test_metrics}
    if config['perform_cv']:
        metrics['cv'] = {
            'r2_mean': r2s.mean(), 'r2_std': r2s.std(),
            'rmse_mean': rmses.mean(), 'rmse_std': rmses.std(),
            'r2_scores': r2s.tolist(),
            'well_r2_per_fold': well_r2_per_fold.tolist(),
            'well_r2_mean': float(np.nanmean(well_r2_per_fold)),
            'well_r2_std': float(np.nanstd(well_r2_per_fold)),
        }
    pipeline['metrics'] = metrics

    pipeline['arrays'] = {
        'X_train_final': X_train_final,
        'X_test_final':  X_test_final,
        'y_train_orig':  y_train_orig.values,
        'y_test_orig':   y_test_orig.values,
        'well_train':    well_train,
        'well_test':     well_test,
    }

    # The critical save
    joblib.dump(pipeline, os.path.join(save_dirs['models'], 'complete_pipeline.pkl'))
    joblib.dump(config,   os.path.join(save_dirs['run'], 'config.pkl'))
    joblib.dump(metrics,  os.path.join(save_dirs['run'], 'metrics.pkl'))

    # Individual artifacts (backward compat)
    joblib.dump(model,                     os.path.join(save_dirs['models'], 'xgboost_model.pkl'))
    joblib.dump(pipeline['imputers'],      os.path.join(save_dirs['models'], 'imputers.pkl'))
    joblib.dump(pipeline['scalers'],       os.path.join(save_dirs['models'], 'scalers.pkl'))
    joblib.dump(pipeline['encoders'],      os.path.join(save_dirs['models'], 'encoders.pkl'))
    joblib.dump(model_rfe,                 os.path.join(save_dirs['models'], 'xgboost_rfe_model.pkl'))
    joblib.dump(importance_df,             os.path.join(save_dirs['dataframes'], 'feature_importance.pkl'))
    joblib.dump(importance_grouped,        os.path.join(save_dirs['dataframes'], 'feature_importance_grouped.pkl'))

    # CV fold-level artifacts
    if fold_importances is not None:
        joblib.dump({
            'fold_importances': fold_importances,
            'feature_names': all_feature_names,
            'cv_imp_lookup': cv_imp_lookup,
            'well_r2_per_fold': well_r2_per_fold,
            'r2_per_fold': r2s,
            'rmse_per_fold': rmses,
            'mae_per_fold': maes,
        }, os.path.join(save_dirs['dataframes'], 'cv_fold_results.pkl'))

    joblib.dump({
        'selected_num_features': selected_features,
        'dropped_num_features': dropped_features,
        'selected_num_mask': selected_mask,
        'rfe_feature_names': rfe_feature_names,
        'rfe_train_metrics': rfe_train_metrics,
        'rfe_test_metrics': rfe_test_metrics,
    }, os.path.join(save_dirs['dataframes'], 'rfe_results.pkl'))

    # UQ artifacts
    joblib.dump({
        'method': 'split_conformal',
        'conformal_radii': conformal_radii,
        'conformal_radii_rfe': conformal_radii_rfe,
        'conformal_levels': uq_levels,
        'n_calib': n_calib,
        'calib_residual_stats': {
            'mean': float(np.mean(calib_residuals)),
            'median': float(np.median(calib_residuals)),
            'std': float(np.std(calib_residuals)),
        },
    }, os.path.join(save_dirs['models'], 'uncertainty_artifacts.pkl'))

    print(f"\n✓ Pipeline saved to {save_dirs['models']}/complete_pipeline.pkl")
    print(f"✓ UQ method: Split Conformal Prediction")
    print(f"  90% CI: prediction × {np.exp(conformal_radii[0.90]):.2f} / ÷ {np.exp(conformal_radii[0.90]):.2f}")
    if config['perform_cv']:
        wr2_clean = well_r2_per_fold[~np.isnan(well_r2_per_fold)]
        print(f"✓ CV ({config['cv_folds']} folds):")
        print(f"  Sample-level R²:    {r2s.mean():.4f} ± {r2s.std():.4f}")
        if len(wr2_clean) > 0:
            print(f"  Well-level med R²:  {wr2_clean.mean():.4f} ± {wr2_clean.std():.4f}")

    return pipeline, save_dirs