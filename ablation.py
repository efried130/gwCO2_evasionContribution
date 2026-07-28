"""
ablation.py
===========
General-purpose ablation test for the CO2 XGBoost pipeline.

Remove any feature group and measure the impact on test metrics.
All conditions share the same well-grouped train/test split.

Usage
-----
    from ablation import run_ablation

    # Ablations based on top-20 features, grouped thematically
    ablations = top20_ablations()
    results = run_ablation(df_cleaned, ablations)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

from train_pipeline_aq_uncertainty2 import (
    CONFIG, NUMERIC_FEATURES, INTERACTION_DEFS,
    add_temporal_features, create_interaction_terms,
    impute_numeric, scale_numeric, encode_categoricals,
    calculate_comprehensive_metrics, print_comprehensive_metrics,
)


# ============================================================================
# TOP-20 ABLATIONS (thematically grouped)
# ============================================================================

def top20_ablations():
    """
    Ablation conditions based on the top-20 feature importance list,
    grouped by theme. Each condition removes one thematic group.

    Top 20 (for reference):
       1. fec_cl_smj_target       0.2503    (categorical — fecal coliform class)
       2. swc_pc_s                0.0885    (soil water content)
       3. HYGEO2_target           0.0326    (categorical — hydrogeology)
       4. arid_organic_index      0.0172    (interaction: soc × aridity)
       5. slt_pc_sav              0.0169    (silt %)
       6. soc_th_sav              0.0164    (soil organic carbon)
       7. snd_pc_sav              0.0158    (sand %)
       8. hydraulic_gradient      0.0155    (interaction: slope × upstream area)
       9. ele_mt_sav              0.0141    (elevation)
      10. slp_dg_sav              0.0133    (slope)
      11. elevation_soil_carbon   0.0131    (interaction: elev × soc)
      12. shallow_organic_index   0.0131    (interaction: soc / dtb)
      13. temp_moisture           0.0119    (interaction: t2m × swc)
      14. ari_ix_sav              0.0110    (aridity index)
      15. year                    0.0109    (temporal)
      16. for_pc_sse              0.0105    (forest %)
      17. DIST_MAIN               0.0105    (distance to main channel)
      18. hdi_ix_sav              0.0104    (human development index)
      19. hft_ix_s09              0.0104    (human footprint 2009)
      20. gdp_ud_sav              0.0104    (GDP)
    """
    return {
        # ── Single-feature ablations for the big hitters ──
        'no_fec_class':     ['fec_cl_smj'],
        'no_swc':           ['swc_pc_s'],
        'no_HYGEO2':        ['HYGEO2'],

        # ── Thematic groups ──
        'no_soil_texture':  ['slt_pc_sav', 'snd_pc_sav', 'cly_pc_sav'],
        'no_soil_organic':  ['soc_th_sav', 'arid_organic_index',
                             'elevation_soil_carbon', 'shallow_organic_index',
                             'depth_weighted_soc'],
        'no_topography':    ['ele_mt_sav', 'slp_dg_sav', 'hydraulic_gradient',
                             'coastal_plain_index'],
        'no_climate':       ['ari_ix_sav', 'swc_pc_s', 'temp_moisture',
                             'temp_moisture_sq'],
        'no_interactions':  ['arid_organic_index', 'hydraulic_gradient',
                             'elevation_soil_carbon', 'shallow_organic_index',
                             'temp_moisture', 'temp_moisture_sq',
                             'depth_weighted_soc', 'swc_proximity_sink',
                             'coastal_plain_index', 'distance_drainage',
                             'elevation_erosion'],
        'no_human':         ['hdi_ix_sav', 'hft_ix_s09', 'gdp_ud_sav'],
        'no_hydro':         ['DIST_MAIN', 'DIST_SINK'],
        'no_landcover':     ['for_pc_sse', 'crp_pc_sse', 'urb_pc_sse',
                             'pst_pc_sse'],
        'no_temporal':      ['year', 'season'],

        # ── Categoricals together ──
        'no_all_categoricals': ['fec_cl_smj', 'HYGEO2', 'lit_cl_smj'],
    }


# ============================================================================
# SHARED DATA PREP
# ============================================================================

def prepare_shared_data(df_cleaned, config=CONFIG):
    """
    Clean + split once. Every ablation condition reuses the same rows
    and the same train/test well partition.
    """
    print("Preparing shared data...")
    df = add_temporal_features(df_cleaned.copy(), 'Date')
    df = create_interaction_terms(df)

    y_original = df[config['label']].copy()
    y = np.log(y_original)
    well_ids = df[config['well_column']].copy()

    # ── Filters (mirrors train_pipeline.py) ──
    mask = y.notna()
    df, y, y_original, well_ids = [
        a[mask].reset_index(drop=True) for a in [df, y, y_original, well_ids]]

    if config['threshold_co2'] is not None:
        m = y_original >= config['threshold_co2']
        df, y, y_original, well_ids = [
            a[m].reset_index(drop=True) for a in [df, y, y_original, well_ids]]

    if config['threshold_co2_high'] is not None:
        m = y_original <= config['threshold_co2_high']
        df, y, y_original, well_ids = [
            a[m].reset_index(drop=True) for a in [df, y, y_original, well_ids]]

    if config.get('filter_by_location_count', False):
        loc_counts = well_ids.value_counts()
        keep = loc_counts[loc_counts >= config['min_samples_per_location']].index
        m = well_ids.isin(keep)
        df, y, y_original, well_ids = [
            a[m].reset_index(drop=True) for a in [df, y, y_original, well_ids]]

    all_feats = list(NUMERIC_FEATURES)
    X_check = df[all_feats].copy()
    missing_per_row = X_check.isnull().sum(axis=1)
    bad = np.zeros(len(df), dtype=bool)
    if config['remove_rows_with_all_na']:
        bad |= (missing_per_row == len(all_feats))
    if config['remove_rows_with_most_na']:
        bad |= (missing_per_row >= int(len(all_feats) * config['most_na_threshold']))
    if config['na_removal_threshold'] is not None:
        bad |= (missing_per_row > int(len(all_feats) * config['na_removal_threshold']))
    df, y, y_original, well_ids = [
        a[~bad].reset_index(drop=True) for a in [df, y, y_original, well_ids]]

    if config['outlier_removal_method'] == 'iqr':
        X_out = df[all_feats]
        outlier_counts = np.zeros(len(df))
        for feat in all_feats:
            fdata = X_out[feat].dropna()
            if len(fdata) < 10:
                continue
            q25, q75 = np.percentile(fdata, [25, 75])
            iqr = q75 - q25
            outlier_counts += (
                (X_out[feat] < q25 - config['outlier_threshold'] * iqr) |
                (X_out[feat] > q75 + config['outlier_threshold'] * iqr)
            ).astype(int)
        outlier_mask = outlier_counts >= config['min_features_for_outlier']
        df, y, y_original, well_ids = [
            a[~outlier_mask].reset_index(drop=True)
            for a in [df, y, y_original, well_ids]]

    # ── Well-grouped split ──
    well_stats = (pd.DataFrame({'well_id': well_ids, 'co2': y})
                  .groupby('well_id')['co2'].median().reset_index())
    well_stats.columns = ['well_id', 'median_co2']
    well_stats['co2_bin'] = pd.qcut(
        well_stats['median_co2'], q=5, labels=False, duplicates='drop')

    train_wells, test_wells = train_test_split(
        well_stats['well_id'], test_size=config['test_size'],
        stratify=well_stats['co2_bin'], random_state=config['random_state'])

    train_mask = well_ids.isin(train_wells)
    test_mask = well_ids.isin(test_wells)

    print(f"  {len(df):,} samples  |  "
          f"{train_mask.sum():,} train ({len(train_wells)} wells)  |  "
          f"{test_mask.sum():,} test ({len(test_wells)} wells)")

    return {
        'df': df, 'y': y, 'y_original': y_original,
        'well_ids': well_ids, 'train_mask': train_mask, 'test_mask': test_mask,
        'config': config,
    }


# ============================================================================
# TRAIN + EVALUATE ONE CONDITION
# ============================================================================

def run_condition(shared, name, remove_features, config):
    """
    Train + evaluate with `remove_features` excluded.

    Handles both numeric and categorical removals:
    if a feature in remove_features is in config['categorical_features'],
    it's dropped from the categorical encoder too.
    """
    df         = shared['df']
    y          = shared['y']
    y_original = shared['y_original']
    well_ids   = shared['well_ids']
    train_mask = shared['train_mask']
    test_mask  = shared['test_mask']

    remove_set = set(remove_features)

    # Numeric features for this condition
    num_features = [f for f in NUMERIC_FEATURES if f not in remove_set and f in df.columns]

    # Categorical features for this condition
    cat_cols = [c for c in config['categorical_features']
                if c in df.columns and c not in remove_set]

    # Splits
    X_train_num = df.loc[train_mask, num_features].reset_index(drop=True)
    X_test_num  = df.loc[test_mask,  num_features].reset_index(drop=True)
    y_train      = y[train_mask].reset_index(drop=True)
    y_test       = y[test_mask].reset_index(drop=True)
    y_train_orig = y_original[train_mask].reset_index(drop=True)
    y_test_orig  = y_original[test_mask].reset_index(drop=True)
    well_train   = well_ids[train_mask].reset_index(drop=True)
    well_test    = well_ids[test_mask].reset_index(drop=True)

    # Impute + scale
    X_train_imp, imputers = impute_numeric(X_train_num, {}, num_features, fit=True)
    X_test_imp, _         = impute_numeric(X_test_num, imputers, num_features, fit=False)
    X_train_scl, scalers  = scale_numeric(X_train_imp, {}, num_features, fit=True)
    X_test_scl, _         = scale_numeric(X_test_imp, scalers, num_features, fit=False)

    # Categoricals
    cond_config = {**config, 'categorical_features': cat_cols}
    X_train_cat = df.loc[train_mask, cat_cols].reset_index(drop=True) if cat_cols else pd.DataFrame()
    X_test_cat  = df.loc[test_mask,  cat_cols].reset_index(drop=True) if cat_cols else pd.DataFrame()
    X_train_cat_enc, encoders, enc_names = encode_categoricals(
        X_train_cat, {}, cond_config, y_train=y_train, fit=True)
    X_test_cat_enc, _, _ = encode_categoricals(
        X_test_cat, encoders, cond_config, fit=False)

    if X_train_cat_enc.size > 0:
        X_train_final = np.hstack([X_train_scl.values, X_train_cat_enc])
        X_test_final  = np.hstack([X_test_scl.values,  X_test_cat_enc])
    else:
        X_train_final = X_train_scl.values
        X_test_final  = X_test_scl.values

    # Train
    model = XGBRegressor(**config['xgb_params'])
    model.fit(X_train_final, y_train)

    # Evaluate
    train_pred = np.exp(model.predict(X_train_final))
    test_pred  = np.exp(model.predict(X_test_final))
    train_m = calculate_comprehensive_metrics(y_train_orig.values, train_pred)
    test_m  = calculate_comprehensive_metrics(y_test_orig.values,  test_pred)

    # CV (skip if disabled for speed)
    cv_r2s = np.array([np.nan])
    if config.get('perform_cv', True):
        gkfold = GroupKFold(n_splits=config['cv_folds'])
        cv_r2s = []
        for ti, vi in gkfold.split(X_train_final, y_train, groups=well_train):
            fm = XGBRegressor(**config['xgb_params'])
            fm.fit(X_train_final[ti], y_train.iloc[ti])
            cv_r2s.append(r2_score(
                np.exp(y_train.iloc[vi].values),
                np.exp(fm.predict(X_train_final[vi]))))
        cv_r2s = np.array(cv_r2s)

    # Well-level metrics on test set
    MIN_SAMPLES_WELL = 10
    test_results_df = pd.DataFrame({
        'MonitoringLocation': well_test.values,
        'CO2_actual': y_test_orig.values,
        'CO2_predicted': test_pred,
    })
    well_metrics_list = []
    for well_id, group in test_results_df.groupby('MonitoringLocation'):
        yt = group['CO2_actual'].values
        yp = group['CO2_predicted'].values
        if len(yt) < MIN_SAMPLES_WELL:
            continue
        wm = calculate_comprehensive_metrics(yt, yp)
        wm['well_id'] = well_id
        wm['n_samples'] = len(yt)
        well_metrics_list.append(wm)
    well_metrics_df = pd.DataFrame(well_metrics_list)

    # Aggregate well-level stats
    well_agg = {}
    if len(well_metrics_df) > 0:
        for metric in ['r2', 'rmsle', 'kge', 'nse', 'relative_error', 'mape']:
            vals = well_metrics_df[metric].dropna().values
            if len(vals) > 0:
                well_agg[metric] = {
                    'median': float(np.median(vals)),
                    'mean': float(np.mean(vals)),
                    'p25': float(np.percentile(vals, 25)),
                    'p75': float(np.percentile(vals, 75)),
                    'n_wells': len(vals),
                }

    return {
        'condition':     name,
        'removed':       list(remove_set),
        'n_num':         len(num_features),
        'n_cat':         len(cat_cols),
        'n_total':       X_train_final.shape[1],
        'train_metrics': train_m,
        'test_metrics':  test_m,
        'cv_r2_mean':    cv_r2s.mean(),
        'cv_r2_std':     cv_r2s.std(),
        'well_metrics':  well_metrics_df,
        'well_agg':      well_agg,
    }


# ============================================================================
# MAIN
# ============================================================================

def run_ablation(df_cleaned, ablations, config=CONFIG):
    """
    Parameters
    ----------
    df_cleaned : DataFrame
        Same input you pass to train().
    ablations : dict
        {condition_name: [list of features to remove]}
        Example:
            {'no_cropland': ['crp_pc_sse', 'crp_per_aridity'],
             'no_HYGEO2':   ['HYGEO2']}

    Returns
    -------
    list of result dicts, one per condition (including FULL baseline).
    """
    shared = prepare_shared_data(df_cleaned, config)

    # Always run baseline first
    all_conditions = {'FULL': []}
    all_conditions.update(ablations)

    results = []
    for name, remove_list in all_conditions.items():
        print(f"\n{'='*70}")
        if remove_list:
            print(f"CONDITION: {name}  (removing {remove_list})")
        else:
            print(f"CONDITION: {name}  (baseline — all features)")
        print(f"{'='*70}")

        res = run_condition(shared, name, remove_list, config)
        results.append(res)

        print_comprehensive_metrics(res['test_metrics'], f"{name} Test")
        print(f"  CV R²: {res['cv_r2_mean']:.4f} ± {res['cv_r2_std']*2:.4f}")
        print(f"  Features: {res['n_total']} ({res['n_num']} numeric + {res['n_cat']} categorical)")

    # ── Summary table ──
    key_metrics = ['r2', 'rmsle', 'kge', 'nse', 'relative_error']
    baseline = results[0]['test_metrics']

    print(f"\n{'='*100}")
    print("ABLATION SUMMARY  (test set)")
    print(f"{'='*100}")
    header = f"{'Condition':<20} {'n_feat':<8} {'CV R²':<14} "
    header += " ".join(f"{m.upper():<14}" for m in key_metrics)
    print(header)
    print("-" * 100)

    for res in results:
        t = res['test_metrics']
        row = f"{res['condition']:<20} {res['n_total']:<8} "
        row += f"{res['cv_r2_mean']:.4f}±{res['cv_r2_std']:.4f}  "
        for m in key_metrics:
            row += f"{t[m]:<14.4f} "
        print(row)

    # ── Deltas vs baseline ──
    print(f"\n{'─'*80}")
    print("DELTAS vs FULL baseline   (positive R²/KGE/NSE = removing hurt;  negative error = removing helped)")
    print(f"{'─'*80}")
    print(f"{'Condition':<20} ", end="")
    print(" ".join(f"{'Δ'+m.upper():<14}" for m in key_metrics))
    print("-" * 80)

    higher_is_better = {'r2', 'kge', 'nse'}

    for res in results[1:]:
        t = res['test_metrics']
        row = f"{res['condition']:<20} "
        for m in key_metrics:
            delta = t[m] - baseline[m]
            # For R²/KGE/NSE: negative delta means removing hurt (feature was useful)
            # For error metrics: positive delta means removing hurt
            if m in higher_is_better:
                marker = '✗' if delta < -0.001 else ('✓' if delta > 0.001 else '~')
            else:
                marker = '✗' if delta > 0.001 else ('✓' if delta < -0.001 else '~')
            row += f"{delta:+.4f} {marker}    "
        print(row)

    print(f"\n  ✗ = removing hurt (feature was useful)")
    print(f"  ✓ = removing helped (feature was noise)")
    print(f"  ~ = negligible difference (<0.001)")

    # ── Well-level summary ──
    well_metrics_keys = ['r2', 'rmsle', 'kge', 'nse', 'relative_error']
    baseline_well = results[0].get('well_agg', {})

    if baseline_well:
        print(f"\n{'='*110}")
        print("WELL-LEVEL ABLATION SUMMARY  (test wells, median metrics)")
        print(f"{'='*110}")
        header = f"{'Condition':<20} {'n_wells':<9} "
        header += " ".join(f"{'med_'+m.upper():<14}" for m in well_metrics_keys)
        print(header)
        print("-" * 110)

        for res in results:
            wa = res.get('well_agg', {})
            n_wells = wa[well_metrics_keys[0]]['n_wells'] if well_metrics_keys[0] in wa else 0
            row = f"{res['condition']:<20} {n_wells:<9} "
            for m in well_metrics_keys:
                if m in wa:
                    row += f"{wa[m]['median']:<14.4f} "
                else:
                    row += f"{'N/A':<14} "
            print(row)

        # Well-level deltas
        print(f"\n{'─'*110}")
        print("WELL-LEVEL DELTAS vs FULL  (median metrics)")
        print(f"{'─'*110}")
        print(f"{'Condition':<20} ", end="")
        print(" ".join(f"{'Δmed_'+m.upper():<14}" for m in well_metrics_keys))
        print("-" * 110)

        for res in results[1:]:
            wa = res.get('well_agg', {})
            row = f"{res['condition']:<20} "
            for m in well_metrics_keys:
                if m in wa and m in baseline_well:
                    delta = wa[m]['median'] - baseline_well[m]['median']
                    if m in higher_is_better:
                        marker = '✗' if delta < -0.002 else ('✓' if delta > 0.002 else '~')
                    else:
                        marker = '✗' if delta > 0.002 else ('✓' if delta < -0.002 else '~')
                    row += f"{delta:+.4f} {marker}    "
                else:
                    row += f"{'N/A':<14} "
            print(row)

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ablation.py <path_to_cleaned_data.parquet>")
        sys.exit(1)

    data_path = sys.argv[1]
    print(f"Loading: {data_path}")
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)

    results = run_ablation(df, top20_ablations())