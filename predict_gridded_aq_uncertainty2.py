"""
predict_gridded.py
==================
Batch prediction on gridded ERA5 (daily + monthly) + GLDAS + HydroATLAS
parquet data using the pipeline saved by train_pipeline.py.

Supports split conformal uncertainty intervals (--uncertainty flag).
"""

import numpy as np
import pandas as pd
import joblib
import os
import gc
import glob
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# ERA5 MONTHLY PARQUET RENAME                                        
# Parquet files have: d2m, t2m, stl1, ...
# Training expects:   d2m_month, t2m_month, stl1_month, ...
# ============================================================================

ERA5_MONTH_RENAME = {                                                
    'd2m': 'd2m_month',       't2m': 't2m_month',                   
    'stl1': 'stl1_month',     'stl2': 'stl2_month',                
    'stl3': 'stl3_month',     'stl4': 'stl4_month',                 
    'snowc': 'snowc_month',   'rsn': 'rsn_month',                   
    'sde': 'sde_month',       'sd': 'sd_month',                     
    'sf': 'sf_month',         'smlt': 'smlt_month',                  
    'tsn': 'tsn_month',       'src': 'src_month',                   
    'swvl1': 'swvl1_month',   'swvl2': 'swvl2_month',               
    'swvl3': 'swvl3_month',   'swvl4': 'swvl4_month',               
    'fal': 'fal_month',       'slhf': 'slhf_month',                 
    'ssr': 'ssr_month',       'str': 'str_month',                   
    'sshf': 'sshf_month',     'ssrd': 'ssrd_month',                 
    'strd': 'strd_month',     'evabs': 'evabs_month',               
    'evaow': 'evaow_month',   'evatc': 'evatc_month',               
    'evavt': 'evavt_month',   'pev': 'pev_month',                   
    'ro': 'ro_month',         'es': 'es_month',                     
    'ssro': 'ssro_month',     'sro': 'sro_month',                   
    'e': 'e_month',           'sp': 'sp_month',                     
    'tp': 'tp_month',         'lai_hv': 'lai_hv_month',             
    'lai_lv': 'lai_lv_month',                                       
}                                                                   

# ============================================================================
# COLUMN NAME MAPPING  (parquet column → training feature name)
# ============================================================================

PARQUET_TO_TRAIN_MAP = {
    # ── Daily ERA5 ──
    'sp': 'sp', 'tp': 'tp', 'pev': 'pev',
    'stl1': 'stl1', 'stl2': 'stl2', 'stl3': 'stl3', 'stl4': 'stl4',
    'swvl1': 'swvl1', 'swvl2': 'swvl2', 'swvl3': 'swvl3', 'swvl4': 'swvl4',
    'fal': 'fal', 'ssrd': 'ssrd', 'strd': 'strd',
    'sshf': 'sshf', 'slhf': 'slhf', 'ssr': 'ssr',
    'evavt': 'evavt', 'evatc': 'evatc', 'evabs': 'evabs',
    't2m': 't2m', 'd2m': 'd2m', 'e': 'e',
    # ── Monthly ERA5  (identity — parquet names already match) ──
    'd2m_month': 'd2m_month', 't2m_month': 't2m_month',
    'stl1_month': 'stl1_month', 'stl2_month': 'stl2_month',
    'stl3_month': 'stl3_month', 'stl4_month': 'stl4_month',
    'snowc_month': 'snowc_month', 'rsn_month': 'rsn_month',
    'sde_month': 'sde_month', 'sd_month': 'sd_month',
    'sf_month': 'sf_month', 'smlt_month': 'smlt_month',
    'tsn_month': 'tsn_month', 'src_month': 'src_month',
    'swvl1_month': 'swvl1_month', 'swvl2_month': 'swvl2_month',
    'swvl3_month': 'swvl3_month', 'swvl4_month': 'swvl4_month',
    'fal_month': 'fal_month', 'slhf_month': 'slhf_month',
    'ssr_month': 'ssr_month', 'str_month': 'str_month',
    'sshf_month': 'sshf_month', 'ssrd_month': 'ssrd_month',
    'strd_month': 'strd_month', 'evabs_month': 'evabs_month',
    'evaow_month': 'evaow_month', 'evatc_month': 'evatc_month',
    'evavt_month': 'evavt_month', 'pev_month': 'pev_month',
    'ro_month': 'ro_month', 'es_month': 'es_month',
    'ssro_month': 'ssro_month', 'sro_month': 'sro_month',
    'e_month': 'e_month', 'sp_month': 'sp_month',
    'tp_month': 'tp_month', 'lai_hv_month': 'lai_hv_month',
    'lai_lv_month': 'lai_lv_month',
    # ── GLDAS ──
    'AvgSurfT_inst': 'AvgSurfT_inst', 'ESoil_tavg': 'ESoil_tavg',
    'Evap_tavg': 'Evap_tavg', 'PotEvap_tavg': 'PotEvap_tavg',
    'Qair_f_inst': 'Qair_f_inst', 'Rainf_f_tavg': 'Rainf_f_tavg',
    'Rainf_tavg': 'Rainf_tavg', 'RootMoist_inst': 'RootMoist_inst',
    'SWE_inst': 'SWE_inst', 'SnowDepth_inst': 'SnowDepth_inst',
    'SoilMoi0_10cm_inst': 'SoilMoi0_10cm_inst',
    'SoilMoi100_200cm_inst': 'SoilMoi100_200cm_inst',
    'SoilTMP0_10cm_inst': 'SoilTMP0_10cm_inst',
    'SoilTMP100_200_inst': 'SoilTMP100_200_inst',
    'Tair_f_inst': 'Tair_f_inst', 'Tveg_tavg': 'Tveg_tavg',
    # ── HydroATLAS static ──
    'DIST_MAIN': 'DIST_MAIN', 'Dd': 'Dd',
    'swc_pc_s': 'swc_pc_s', 'ari_ix_sav': 'ari_ix_sav',
    'dtb': 'dtb', 'ero_kh_sav': 'ero_kh_sav',
    'ele_mt_sav': 'ele_mt_sav', 'soc_th_sav': 'soc_th_sav',
}

# ============================================================================
# MONTHLY VARIABLE PIVOT  (HydroATLAS monthly → single column per month)
# ============================================================================

MONTHLY_VARS = {
    'swc_pc_s': 'swc_pc_s',
    'aet_mm_s': 'aet_mm_s',
    'cmi_ix_s': 'cmi_ix_s',
    'pet_mm_s': 'pet_mm_s',
    'pre_mm_s': 'pre_mm_s',
    'snw_pc_s': 'snw_pc_s',
    'tmp_dc_s': 'tmp_dc_s',
}

def pivot_monthly_hydro(df_hydro, month):
    """Select the correct month column for each HydroATLAS monthly variable."""
    drop_cols = []
    for train_name, base in MONTHLY_VARS.items():
        month_col = f"{base}{month:02d}"
        if month_col in df_hydro.columns:
            df_hydro[train_name] = df_hydro[month_col]
        else:
            df_hydro[train_name] = np.nan
        for m in range(1, 13):
            col = f"{base}{m:02d}"
            if col in df_hydro.columns:
                drop_cols.append(col)
    drop_cols = list(set(drop_cols))
    df_hydro = df_hydro.drop(columns=drop_cols, errors='ignore')
    return df_hydro


def rename_parquet_to_train(df, mapping=PARQUET_TO_TRAIN_MAP):
    rename_dict = {c: mapping[c] for c in df.columns if c in mapping}
    return df.rename(columns=rename_dict)


# ============================================================================
# LOAD HYDROATLAS  (static — called once)
# ============================================================================

def load_hydro(hydro_dir):
    if not os.path.exists(hydro_dir):
        raise FileNotFoundError(f"HydroATLAS dir not found: {hydro_dir}")

    if os.path.isfile(hydro_dir):
        print(f"  Loading HydroATLAS from file: {hydro_dir}", flush=True)
        return pd.read_parquet(hydro_dir)

    files = sorted(set(
        glob.glob(os.path.join(hydro_dir, "*.parquet")) +
        glob.glob(os.path.join(hydro_dir, "**", "*.parquet"), recursive=True)
    ))

    if not files:
        raise FileNotFoundError(f"No .parquet files found in {hydro_dir}")

    print(f"  Loading HydroATLAS from {len(files)} file(s):", flush=True)
    for f in files:
        print(f"    {os.path.basename(f)}", flush=True)

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.drop_duplicates(subset=['latitude', 'longitude'])

    df['latitude']  = df['latitude'].round(4)
    df['longitude'] = df['longitude'].round(4)

    static_cols = [c for c in df.columns if c not in ['latitude', 'longitude']]
    monthly_found = {}
    for train_name, base in MONTHLY_VARS.items():
        month_cols = [f"{base}{m:02d}" for m in range(1, 13)]
        present = [c for c in month_cols if c in df.columns]
        if present:
            monthly_found[train_name] = len(present)

    print(f"  HydroATLAS: {len(df):,} grid cells, "
          f"{len(static_cols)} total columns", flush=True)
    if monthly_found:
        print(f"  Monthly variables: {monthly_found}", flush=True)
    return df


# ============================================================================
# LOAD OPTIONAL PARQUET DIR  (used for monthly ERA5 and GLDAS)
# ============================================================================

def load_parquet_partition(base_dir, year, month):
    """
    Try to load parquet from year=/month= partition structure.
    Returns DataFrame or None if not found.
    """
    part_path = os.path.join(base_dir, f"year={year}", f"month={month}")
    if not os.path.exists(part_path):
        return None
    try:
        return pd.read_parquet(part_path)
    except Exception as e:
        print(f"  ⚠ Failed to read {part_path}: {e}", flush=True)
        return None


# ============================================================================
# CORE PREDICTION FUNCTION  (single DataFrame)
# ============================================================================

def predict_chunk(df, pipeline, use_rfe_model=False, predict_uncertainty=False):
    """
    Predict CO2 for a single chunk.

    Parameters
    ----------
    df : pd.DataFrame
        Merged ERA5 daily + monthly + GLDAS + HydroATLAS data.
    pipeline : dict
        Complete pipeline from training.
    use_rfe_model : bool
        Use RFE-reduced model.
    predict_uncertainty : bool
        If True, return dict with point predictions + CI columns.

    Returns
    -------
    np.ndarray or dict
    """
    config   = pipeline['config']
    features = pipeline['numeric_features']
    scalers  = pipeline['scalers']
    imputers = pipeline['imputers']
    encoders = pipeline['encoders']
    interaction_defs = pipeline.get('interaction_defs', [])

    if use_rfe_model:
        model = pipeline['model_rfe']
        selected_mask = pipeline['rfe_selected_mask']
    else:
        model = pipeline['model']
        selected_mask = None

    # ── Unwrap RFECV to avoid refitting ───────────────────────────
    from sklearn.feature_selection import RFECV, RFE
    if isinstance(model, (RFECV, RFE)):
        print("    ℹ Unwrapping RFECV → using fitted estimator directly",
              flush=True)
        selected_mask = model.support_
        model = model.estimator_
        if use_rfe_model:
            pipeline['model_rfe'] = model
            pipeline['rfe_selected_mask'] = selected_mask
        else:
            pipeline['model'] = model

    # --- 1. Rename columns ---
    df = rename_parquet_to_train(df.copy())

    # --- 2. Temporal features ---
    if 'valid_time' in df.columns:
        df['valid_time'] = pd.to_datetime(df['valid_time'])
        df['month']       = df['valid_time'].dt.month
        df['day_of_year'] = df['valid_time'].dt.dayofyear
        df['year']        = df['valid_time'].dt.year

    if 'month' in df.columns:
        df['season']    = df['month'].astype(int) % 12 // 3 + 1
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    if 'day_of_year' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    elif 'month' in df.columns:
        approx_doy = (df['month'] - 0.5) * 30.44
        df['day_sin'] = np.sin(2 * np.pi * approx_doy / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * approx_doy / 365.25)

    # --- 3. Interaction terms ---
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

    # --- 4. Extract numeric features ---
    X_num = pd.DataFrame(index=df.index)
    missing_cols = []
    for feat in features:
        if feat in df.columns:
            X_num[feat] = df[feat].values
        else:
            X_num[feat] = np.nan
            missing_cols.append(feat)
    if missing_cols:
        print(f"    ⚠ Missing columns filled with NaN: {missing_cols}",
              flush=True)

    # --- 5. Impute ---
    if 'knn' in imputers:
        X_num = pd.DataFrame(imputers['knn'].transform(X_num),
                             columns=features)
    else:
        for feat, imp in imputers.items():
            if feat in X_num.columns:
                X_num[[feat]] = imp.transform(X_num[[feat]])
    X_num = X_num.fillna(0)

    # --- 6. Scale ---
    for feat, info in scalers.items():
        if feat not in X_num.columns:
            continue
        if info.get('use_log', False):
            shift = info['shift']
            X_num[feat] = np.log(X_num[feat] + shift + 1)
        X_num[[feat]] = info['scaler'].transform(X_num[[feat]])

    # --- 7. Categorical encoding ---
    cat_cols = [c for c in config.get('categorical_features', [])
                if c in df.columns]

    all_cat = config.get('categorical_features', [])
    missing_cat = [c for c in all_cat if c not in df.columns]

    if missing_cat:
        print(f"    ⚠ Missing categorical columns: {missing_cat}",
              flush=True)
    if cat_cols and encoders:
        strategy = config.get('encoding_strategy', {})
        df_cat = df[cat_cols].copy()
        for col in cat_cols:
            df_cat[col] = df_cat[col].astype(str).replace(
                ['nan', 'None', 'NaN', '<NA>'], 'Missing')

        encoded_arrays = []
        ordinal_cols = [c for c in cat_cols
                        if strategy.get(c) == 'ordinal']
        target_cols  = [c for c in cat_cols
                        if strategy.get(c) == 'target']
        onehot_cols  = [c for c in cat_cols
                        if strategy.get(c) == 'onehot']

        for col in ordinal_cols:
            if col in encoders:
                encoded_arrays.append(
                    encoders[col].transform(df_cat[[col]]))
        if target_cols and 'target' in encoders:
            encoded_arrays.append(
                encoders['target'].transform(
                    df_cat[target_cols]).values)
        if onehot_cols and 'onehot' in encoders:
            encoded_arrays.append(
                encoders['onehot'].transform(df_cat[onehot_cols]))

        X_cat = (np.hstack(encoded_arrays) if encoded_arrays
                 else np.array([]).reshape(len(df), 0))
    else:
        X_cat = np.array([]).reshape(len(df), 0)

    # --- 8. Combine & predict ---
    if use_rfe_model and selected_mask is not None:
        X_num_arr = X_num.values[:, selected_mask]
    else:
        X_num_arr = X_num.values

    X_final = (np.hstack([X_num_arr, X_cat]) if X_cat.size > 0
               else X_num_arr)

    y_log  = model.predict(X_final)
    y_pred = np.exp(y_log)

    # ── Return point predictions only (backward compatible) ───
    if not predict_uncertainty:
        return y_pred

    # ── Split conformal intervals ─────────────────────────────
    result = {'CO2_predicted': y_pred.astype(np.float32)}

    if use_rfe_model:
        radii = pipeline.get('rfe_conformal_radii', {})
    else:
        radii = pipeline.get('conformal_radii', {})

    if not radii:
        print("    ⚠ No conformal_radii in pipeline — "
              "returning point predictions only", flush=True)
        return result

    for level, radius in sorted(radii.items()):
        pct = int(level * 100)
        result[f'CI_lower_{pct}'] = np.exp(y_log - radius).astype(np.float32)
        result[f'CI_upper_{pct}'] = np.exp(y_log + radius).astype(np.float32)

    return result


# ============================================================================
# BATCH PREDICTION
# ============================================================================

def predict_gridded(
    pipeline_path=None,
    pipeline=None,
    # ── All four input directories ──
    era5_daily_dir="/nas/cee-water/cjgleason/ellie/co2/data/era5_parquet_01deg",
    era5_month_dir="/nas/cee-water/cjgleason/ellie/co2/data/era5_month_parquet_01deg",
    hydro_dir="/nas/cee-water/cjgleason/ellie/co2/data/hydroatlas_parquet_01deg",
    gldas_dir="/nas/cee-water/cjgleason/ellie/co2/data/gldas_parquet_01deg",
    out_dir="/nas/cee-water/cjgleason/ellie/co2/data/co2_predictions",
    year_range=(1970, 2024),
    use_rfe_model=False,
    predict_uncertainty=False,
    batch_log_every=10,
    # ── Backward compat alias ──
    era5_dir=None,
):
    """
    Predict CO2 for all year/month partitions.

    Loads and merges up to four data sources per timestep:
      1. ERA5 daily      (era5_daily_dir)   — daily reanalysis variables
      2. ERA5 monthly    (era5_month_dir)   — monthly aggregated variables
      3. HydroATLAS      (hydro_dir)        — static + monthly-pivoted
      4. GLDAS           (gldas_dir)        — land-surface model variables
    """

    # ── Backward compatibility: old `era5_dir` kwarg ──
    if era5_dir is not None and era5_daily_dir == \
            "/nas/cee-water/cjgleason/ellie/co2/data/era5_parquet_01deg":
        era5_daily_dir = era5_dir
        print(f"  ⚠ Deprecated `era5_dir` mapped to `era5_daily_dir`",
              flush=True)

    if pipeline is None:
        assert pipeline_path is not None, \
            "Provide pipeline_path or pipeline"
        print(f"Loading pipeline from {pipeline_path}...", flush=True)
        pipeline = joblib.load(pipeline_path)

    os.makedirs(out_dir, exist_ok=True)

    model_label = "RFE model" if use_rfe_model else "full model"
    print(f"\n{'='*80}", flush=True)
    print(f"GRIDDED CO2 PREDICTION  ({model_label})", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  ERA5 daily input:   {era5_daily_dir}", flush=True)
    print(f"  ERA5 monthly input: {era5_month_dir}", flush=True)
    print(f"  HydroATLAS input:   {hydro_dir}", flush=True)
    print(f"  GLDAS input:        {gldas_dir}", flush=True)
    print(f"  Output:             {out_dir}", flush=True)
    print(f"  Years:              {year_range[0]}–{year_range[1]}",
          flush=True)
    print(f"  Numeric features:   "
          f"{len(pipeline['numeric_features'])}", flush=True)

    if predict_uncertainty:
        uq_method = pipeline.get('uncertainty_method', 'split_conformal')
        uq_levels = pipeline.get('conformal_levels', [])
        if use_rfe_model:
            radii = pipeline.get('rfe_conformal_radii', {})
        else:
            radii = pipeline.get('conformal_radii', {})
        print(f"  Uncertainty:        Split conformal at "
              f"{[int(l*100) for l in uq_levels]}%", flush=True)
        for level, radius in sorted(radii.items()):
            pct = int(level * 100)
            print(f"    {pct}%: ±{radius:.3f} log-space "
                  f"(×{np.exp(radius):.2f} / ÷{np.exp(radius):.2f})",
                  flush=True)
    else:
        print(f"  Uncertainty:        OFF", flush=True)

    # ── Check which directories actually exist ────────────────────
    dir_status = {
        'era5_daily':  os.path.exists(era5_daily_dir),
        'era5_month':  (era5_month_dir is not None and
                        os.path.exists(era5_month_dir)),
        'hydro':       os.path.exists(hydro_dir),
        'gldas':       (gldas_dir is not None and
                        os.path.exists(gldas_dir)),
    }
    for name, exists in dir_status.items():
        status = "✓ found" if exists else "✗ NOT FOUND"
        print(f"    {name:<15} {status}", flush=True)

    if not dir_status['era5_daily'] and not dir_status['era5_month']:
        raise FileNotFoundError(
            "Neither ERA5 daily nor monthly directory found — "
            "cannot proceed.")

    # ── Load HydroATLAS ONCE ──────────────────────────────────────
    print(f"\nLoading HydroATLAS (static + monthly)...", flush=True)
    df_hydro_full = load_hydro(hydro_dir)

    # ── Rename GWScapes columns to match training feature names ───
    gwscape_renames = {
        'gwscape_Irrigation': 'Irrigation',
        'gwscape_Climate_Coupling': 'Climate_Coupling',
        'gwscape_Field_Size': 'Field_Size',
        'gwscape_GDE_Aquatic': 'GDE_Aquatic',
        'gwscape_GDE_Terrestrial': 'GDE_Terrestrial',
        'gwscape_Storage_Capacity': 'Storage_Capacity',
        'gwscape_Government_Effectiveness': 'Government_Effectiveness',
        'gwscape_Access_Improved_Drinking_Water': 'Access_Improved_Drinking_Water',
    }
    renames_found = {k: v for k, v in gwscape_renames.items() 
                     if k in df_hydro_full.columns}
    if renames_found:
        df_hydro_full = df_hydro_full.rename(columns=renames_found)
        print(f"  Renamed {len(renames_found)} GWScapes columns: "
              f"{list(renames_found.values())}", flush=True)

    for train_name, base in MONTHLY_VARS.items():
        present = sum(1 for m in range(1, 13)
                      if f"{base}{m:02d}" in df_hydro_full.columns)
        if present > 0:
            print(f"  {train_name}: {present}/12 monthly columns found",
                  flush=True)

    # ── Process year/month ────────────────────────────────────────
    print(f"\nProcessing predictions...\n", flush=True)

    start_time   = datetime.now()
    total_rows   = 0
    total_chunks = 0
    skipped      = 0

    for year in range(year_range[0], year_range[1] + 1):
        for month in range(1, 13):

            out_part_dir = os.path.join(
                out_dir, f"year={year}", f"month={month}")
            out_path = os.path.join(out_part_dir, "predictions.parquet")

            if os.path.exists(out_path):
                skipped += 1
                continue

            # ── Load ERA5 daily ───────────────────────────────────
            df_era5_daily = None
            if dir_status['era5_daily']:
                era5d_path = os.path.join(
                    era5_daily_dir,
                    f"year={year}", f"month={month}")
                if os.path.exists(era5d_path):
                    try:
                        df_era5_daily = pd.read_parquet(era5d_path)
                    except Exception as e:
                        print(f"  ⚠ ERA5 daily {year}-{month:02d}: {e}",
                              flush=True)

            # ── Load ERA5 monthly ─────────────────────────────────
            df_era5_month = None
            if dir_status['era5_month']:
                df_era5_month = load_parquet_partition(
                    era5_month_dir, year, month)
                if df_era5_month is not None and not df_era5_month.empty:
                    df_era5_month = df_era5_month.rename(
                        columns=ERA5_MONTH_RENAME)

            # ── Load GLDAS ────────────────────────────────────────
            df_gldas = None
            if dir_status['gldas']:
                df_gldas = load_parquet_partition(
                    gldas_dir, year, month)

            # ── Need at least one ERA5 source ─────────────────────
            if df_era5_daily is None and df_era5_month is None:
                skipped += 1
                continue

            # ── Pick the primary grid (daily preferred) ───────────
            if df_era5_daily is not None and not df_era5_daily.empty:
                df = df_era5_daily.copy()
            elif df_era5_month is not None and not df_era5_month.empty:
                df = df_era5_month.copy()
            else:
                skipped += 1
                continue

            df['latitude']  = df['latitude'].round(4)
            df['longitude'] = df['longitude'].round(4)

            # ── Merge ERA5 monthly (if loaded separately) ─────────
            if (df_era5_month is not None
                    and not df_era5_month.empty
                    and df_era5_daily is not None):
                df_era5_month['latitude'] = \
                    df_era5_month['latitude'].round(4)
                df_era5_month['longitude'] = \
                    df_era5_month['longitude'].round(4)

                month_only_cols = [
                    c for c in df_era5_month.columns
                    if c not in df.columns
                    or c in ['latitude', 'longitude']
                ]
                if ('valid_time' in df.columns
                        and 'valid_time' in df_era5_month.columns):
                    df_era5_month = df_era5_month.drop(
                        columns=['valid_time'], errors='ignore')
                    month_only_cols = [
                        c for c in df_era5_month.columns
                        if c not in df.columns
                        or c in ['latitude', 'longitude']
                    ]

                df = df.merge(
                    df_era5_month[month_only_cols],
                    on=['latitude', 'longitude'],
                    how='left',
                )

            # ── Merge GLDAS ───────────────────────────────────────
            if df_gldas is not None and not df_gldas.empty:
                df_gldas['latitude'] = \
                    df_gldas['latitude'].round(4)
                df_gldas['longitude'] = \
                    df_gldas['longitude'].round(4)
                gldas_only_cols = [
                    c for c in df_gldas.columns
                    if c not in df.columns
                    or c in ['latitude', 'longitude']
                ]
                if 'valid_time' in df_gldas.columns:
                    df_gldas = df_gldas.drop(
                        columns=['valid_time'], errors='ignore')
                    gldas_only_cols = [
                        c for c in df_gldas.columns
                        if c not in df.columns
                        or c in ['latitude', 'longitude']
                    ]
                df = df.merge(
                    df_gldas[gldas_only_cols],
                    on=['latitude', 'longitude'],
                    how='left',
                )

            # ── Merge HydroATLAS (pivoted for this month) ─────────
            df_hydro_month = pivot_monthly_hydro(
                df_hydro_full.copy(), month)

            df = df.merge(
                df_hydro_month,
                on=['latitude', 'longitude'],
                how='left',
            )

            if 'year'  not in df.columns:
                df['year']  = year
            if 'month' not in df.columns:
                df['month'] = month

            n_rows = len(df)

            # ── Predict ───────────────────────────────────────────
            try:
                chunk_result = predict_chunk(
                    df, pipeline,
                    use_rfe_model=use_rfe_model,
                    predict_uncertainty=predict_uncertainty,
                )
            except Exception as e:
                print(f"  ✗ Error predicting {year}-{month:02d}: {e}",
                      flush=True)
                import traceback
                traceback.print_exc()
                skipped += 1
                del df
                if df_era5_daily is not None:
                    del df_era5_daily
                if df_era5_month is not None:
                    del df_era5_month
                if df_gldas is not None:
                    del df_gldas
                del df_hydro_month
                gc.collect()
                continue

            # ── Build output DataFrame ────────────────────────────
            if isinstance(chunk_result, dict):
                out_df = pd.DataFrame({
                    'latitude':      df['latitude'].values,
                    'longitude':     df['longitude'].values,
                    'year':          year,
                    'month':         month,
                    'CO2_predicted': chunk_result['CO2_predicted'],
                })
                for key, vals in chunk_result.items():
                    if key.startswith('CI_'):
                        out_df[key] = vals
            else:
                out_df = pd.DataFrame({
                    'latitude':      df['latitude'].values,
                    'longitude':     df['longitude'].values,
                    'year':          year,
                    'month':         month,
                    'CO2_predicted': chunk_result.astype(np.float32),
                })

            if 'valid_time' in df.columns:
                out_df['valid_time'] = df['valid_time'].values

            os.makedirs(out_part_dir, exist_ok=True)
            out_df.to_parquet(out_path, engine='pyarrow',
                              compression='snappy', index=False)

            total_rows   += n_rows
            total_chunks += 1

            if total_chunks % batch_log_every == 0 or total_chunks <= 3:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(
                    f"  [{total_chunks:>5} chunks] {year}-{month:02d} | "
                    f"{n_rows:>10,} rows | "
                    f"{total_rows:>12,} total | {rate:,.0f} rows/sec",
                    flush=True)

            del df, out_df, chunk_result, df_hydro_month
            if df_era5_daily is not None:
                del df_era5_daily
            if df_era5_month is not None:
                del df_era5_month
            if df_gldas is not None:
                del df_gldas
            gc.collect()

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*80}", flush=True)
    print(f"PREDICTION COMPLETE", flush=True)
    print(f"  Chunks processed: {total_chunks}", flush=True)
    print(f"  Chunks skipped:   {skipped}", flush=True)
    print(f"  Total rows:       {total_rows:,}", flush=True)
    print(f"  Elapsed:          {elapsed/60:.1f} min ({elapsed:.0f} sec)",
          flush=True)
    print(f"  Output:           {out_dir}", flush=True)
    if predict_uncertainty:
        print(f"  Uncertainty:      Split conformal intervals included",
              flush=True)
    print(f"{'='*80}\n", flush=True)

    return out_dir


# ============================================================================
# REASSEMBLE TO NETCDF
# ============================================================================

def predictions_to_netcdf(pred_parquet_dir, out_nc_path,
                          year_range=(1970, 2024)):
    """
    Reassemble partitioned prediction parquets into a single NetCDF.
    Automatically includes any CI_lower_XX / CI_upper_XX columns.
    """
    import xarray as xr

    print(f"Assembling predictions to NetCDF: {out_nc_path}", flush=True)
    all_dfs = []

    for year in range(year_range[0], year_range[1] + 1):
        for month in range(1, 13):
            path = os.path.join(
                pred_parquet_dir, f"year={year}",
                f"month={month}", "predictions.parquet")
            if os.path.exists(path):
                all_dfs.append(pd.read_parquet(path))

    if not all_dfs:
        print("No prediction files found!")
        return

    big_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total rows: {len(big_df):,}", flush=True)

    data_cols = ['CO2_predicted']
    ci_cols = [c for c in big_df.columns if c.startswith('CI_')]
    data_cols.extend(sorted(ci_cols))
    print(f"  Data variables: {data_cols}", flush=True)

    ds = (big_df.set_index(['valid_time', 'latitude', 'longitude'])
                [data_cols].to_xarray())

    ds['CO2_predicted'].attrs['units']     = 'mol/L'
    ds['CO2_predicted'].attrs['long_name'] = \
        'Predicted groundwater CO2(aq)'

    for col in ci_cols:
        ds[col].attrs['units'] = 'mol/L'
        if 'lower' in col:
            pct = col.split('_')[-1]
            ds[col].attrs['long_name'] = \
                f'Split conformal {pct}% confidence interval lower bound'
        elif 'upper' in col:
            pct = col.split('_')[-1]
            ds[col].attrs['long_name'] = \
                f'Split conformal {pct}% confidence interval upper bound'

    ds.to_netcdf(out_nc_path)
    print(f"  Saved to {out_nc_path}", flush=True)
    return ds


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict gridded CO2 from ERA5 + HydroATLAS parquet")
    parser.add_argument("--pipeline",        required=True)
    parser.add_argument("--era5-daily-dir",  required=True,
        help="Daily ERA5 parquets (year=/month= partitions)")
    parser.add_argument("--era5-month-dir",  required=True,
        help="Monthly ERA5 parquets (year=/month= partitions)")
    parser.add_argument("--hydro-dir",       required=True,
        help="HydroATLAS parquets (static)")
    parser.add_argument("--gldas-dir",       default=None,
        help="GLDAS parquets (year=/month= partitions), optional")
    parser.add_argument("--out-dir",         required=True)
    parser.add_argument("--year-start",      type=int, default=1970)
    parser.add_argument("--year-end",        type=int, default=2024)
    parser.add_argument("--use-rfe",         action="store_true")
    parser.add_argument("--uncertainty",     action="store_true",
        help="Include split conformal uncertainty intervals in output")
    parser.add_argument("--to-netcdf",       default=None)
    # ── Backward compat ──
    parser.add_argument("--era5-dir",        default=None,
        help="DEPRECATED — use --era5-daily-dir instead")
    args = parser.parse_args()

    predict_gridded(
        pipeline_path=args.pipeline,
        era5_daily_dir=args.era5_daily_dir,
        era5_month_dir=args.era5_month_dir,
        hydro_dir=args.hydro_dir,
        gldas_dir=args.gldas_dir,
        out_dir=args.out_dir,
        year_range=(args.year_start, args.year_end),
        use_rfe_model=args.use_rfe,
        predict_uncertainty=args.uncertainty,
        era5_dir=args.era5_dir,
    )

    if args.to_netcdf:
        predictions_to_netcdf(
            args.out_dir, args.to_netcdf,
            year_range=(args.year_start, args.year_end))