"""
contribution_saccardi.py

Aggregate Saccardi et al. (2024) NHD-HR reach CO2 fluxes to HUC2 and compute the
groundwater contribution percentage against that stream efflux estimate.
Requires gw_co2_flux_by_huc2_*.csv from gw_flux.py.
"""
co2_predictions = "co2_predictions_aq_uncertainty2"

"""
Aggregate Saccardi et al. (2024) NHD-HR CO2 fluxes to HUC2.
Uses pre-computed per-reach FCO2 from Craig's CONUS_carbon model output.

Each CSV = one HUC4, filename encodes HUC4 ID.
Total reach emission = FCO2_gC_m2_yr × LengthKM × 1000 × W_m

Filter: W_m >= 0.3m (minimum stream width, consistent with Liu & Butman)

Uncertainty:
  Saccardi: published ±23 TgC/yr distributed proportionally to flux.
  Butman stream efflux: ~95% CI rescaled to 90% CI (×1.645/1.960)
  GW CO2 flux: CQR 90% prediction intervals

  SENSITIVITY (diagnostic only, Saccardi Eqs S16-S17): per-HUC4 δF
    from calibration residuals (δpCO2 = (1/cost)/2,
    δF = k_median × δpCO2 × SA_basin).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
from glob import glob
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
tag = 'aq' if co2_predictions.endswith('_aq_uncertainty2') else 'ppm'

SACCARDI_DIR      = Path("data/saccardi_cache")
BASE_DIR          = Path("data")
HUC_PATH          = BASE_DIR / 'shp_files/Watershed_Boundary_Dataset_HUC_2.gpkg'
BUTMAN_PATH       = BASE_DIR / 'runoff/aquatic_carbon_butman.csv'
GW_FLUX_PATH      = BASE_DIR / f'gw_co2_flux_by_huc2_1979_2014_{tag}.csv'
RUNOFF_PATH       = BASE_DIR / 'runoff/era5_land_mean_annual_runoff_1979_2014_nofreeze.nc'
CALIB_PATH        = BASE_DIR / 'runoff/calibratedParameters_all_params.csv'
CLIMATE_JSON_PATH = BASE_DIR / 'runoff/huc2_monthly_climate_1971_2000.json'
OUTPUT_PATH       = BASE_DIR / f'saccardi_stream_co2_by_huc2_{tag}.csv'

CONUS_HUC2  = [str(i).zfill(2) for i in range(1, 19)]
MIN_WIDTH_M = 0.3
TG_PER_G    = 1e-12
SEC_PER_YR  = 365.25 * 24 * 3600

# Published CONUS uncertainty from Saccardi et al. (2024): ±23 TgC/yr
SACCARDI_PUBLISHED_DF = 23.0  # TgC/yr


# ============================================================
# HENRY'S LAW CONSTANT — per-basin from temperature
# ============================================================
def compute_KH(Tw_celsius):
    Tk = Tw_celsius + 273.15
    A, B, C, D, E = 108.3865, 0.01985076, -6919.53, -40.4515, 669365.0
    log10_KH = A + B * Tk + C / Tk + D * np.log10(Tk) + E / (Tk**2)
    return 10**log10_KH


def ppm_to_gCm3(ppm, Tw_celsius):
    KH = compute_KH(Tw_celsius)
    return ppm * 1e-6 * KH * 1000 * 12.01


# ============================================================
# HELPER: Load HUC2 climate data
# ============================================================
def load_huc2_climate(climate_json_path):
    with open(climate_json_path, 'r') as f:
        huc_climate = json.load(f)
    huc2_Tw = {}
    for huc2, clim in huc_climate.items():
        if huc2 not in CONUS_HUC2:
            continue
        mean_Ta = np.mean(clim['temp'])
        Tw = 3.941 + 0.818 * mean_Ta  # Lauerwald et al. 2015
        Tw = max(Tw, 0.1)
        huc2_Tw[huc2] = Tw
    return huc2_Tw


# ============================================================
# HELPER: Load HUC4 calibration parameters (sensitivity diagnostic)
# ============================================================
def load_huc4_calibration(calib_path, huc2_Tw):
    calib = pd.read_csv(calib_path)
    calib['huc4'] = calib['id'].str.replace('calibratedParameters_', '', regex=False)
    calib['huc2'] = calib['huc4'].str[:2]
    calib['fitness_clamped'] = calib['fitness'].clip(lower=1e-10)
    calib['delta_pco2_ppm'] = (1.0 / calib['fitness_clamped']) / 2.0
    calib['Tw'] = calib['huc2'].map(huc2_Tw).fillna(15.0)
    calib['KH'] = calib['Tw'].apply(compute_KH)
    calib['delta_pco2_gCm3'] = calib.apply(
        lambda r: ppm_to_gCm3(r['delta_pco2_ppm'], r['Tw']), axis=1
    )

    print(f"\n  Loaded {len(calib)} HUC4 calibration records")
    print(f"  Fitness range:  {calib['fitness'].min():.6f} – {calib['fitness'].max():.1f}")
    print(f"  δpCO2 range:    {calib['delta_pco2_ppm'].min():.1f} – "
          f"{calib['delta_pco2_ppm'].max():.0f} ppm")
    print(f"  δpCO2 median:   {calib['delta_pco2_ppm'].median():.1f} ppm")

    return calib.set_index('huc4')


# ============================================================
# HELPER: Load HUC2 watershed areas
# ============================================================
def load_huc2_areas(huc_path, conus_huc2):
    hucs = gpd.read_file(huc_path)
    hucs = hucs[hucs['huc2'].isin(conus_huc2)].copy()
    return hucs.set_index('huc2')['areasqkm']


# ============================================================
# STEP 1: Discover CSVs
# ============================================================
def discover_csvs(saccardi_dir):
    csv_files = sorted(glob(str(saccardi_dir / 'final_*.csv')))
    print(f"Found {len(csv_files)} CSV files")
    file_map = []
    for fpath in csv_files:
        fname = Path(fpath).stem
        huc4 = fname.replace('final_', '')
        huc2 = huc4[:2]
        file_map.append({'huc4': huc4, 'huc2': huc2, 'path': fpath})
    file_df = pd.DataFrame(file_map)
    print(f"  HUC2 regions covered: {sorted(file_df['huc2'].unique())}")
    print(f"  HUC4 basins: {len(file_df)}")
    return file_df


# ============================================================
# STEP 2: Process CSVs — aggregate to HUC2
# ============================================================
def process_all_csvs(file_df, huc4_calib):
    print("\nProcessing reach-level CSVs...")

    usecols = ['NHDPlusID', 'CO2_ppm', 'StreamOrde', 'LengthKM',
               'Q_m3_s', 'W_m', 'waterbody', 'FCO2_gC_m2_yr',
               'k_co2_m_s', 'k600_m_s']

    huc2_accum = {}
    huc4_dF_list = []

    for _, row in file_df.iterrows():
        huc4 = row['huc4']
        huc2 = row['huc2']
        fpath = row['path']

        try:
            df = pd.read_csv(fpath, usecols=usecols)
        except ValueError:
            df = pd.read_csv(fpath)
            for col in usecols:
                if col not in df.columns:
                    df[col] = np.nan

        n_total = len(df)
        df = df[df['W_m'] >= MIN_WIDTH_M].copy()
        n_kept = len(df)

        df_rivers = df[df['waterbody'] == 'River'].copy()
        df_lakes  = df[df['waterbody'] != 'River'].copy()

        df_rivers['reach_area_m2'] = df_rivers['LengthKM'] * 1000.0 * df_rivers['W_m']

        if 'lakeSA_m2' in df.columns:
            df_lakes['reach_area_m2'] = pd.to_numeric(
                df_lakes.get('lakeSA_m2', 0), errors='coerce'
            ).fillna(df_lakes['LengthKM'] * 1000.0 * df_lakes['W_m'])
        else:
            df_lakes['reach_area_m2'] = df_lakes['LengthKM'] * 1000.0 * df_lakes['W_m']

        df = pd.concat([df_rivers, df_lakes], ignore_index=True)
        df['reach_flux_gC_yr'] = df['FCO2_gC_m2_yr'] * df['reach_area_m2']

        #  Per-HUC4 calibration-residual δF (DIAGNOSTIC only)
        k_all = df['k_co2_m_s'].dropna()
        k_median = k_all.median() if len(k_all) > 0 else 0.0
        SA_basin = df['reach_area_m2'].sum()

        if huc4 in huc4_calib.index:
            delta_pco2_gCm3 = huc4_calib.loc[huc4, 'delta_pco2_gCm3']
            delta_pco2_ppm  = huc4_calib.loc[huc4, 'delta_pco2_ppm']
            fitness         = huc4_calib.loc[huc4, 'fitness']
            Tw              = huc4_calib.loc[huc4, 'Tw']
            KH              = huc4_calib.loc[huc4, 'KH']
        else:
            delta_pco2_gCm3 = np.nan
            delta_pco2_ppm  = np.nan
            fitness         = np.nan
            Tw              = np.nan
            KH              = np.nan

        dF_gCs = k_median * delta_pco2_gCm3 * SA_basin if not np.isnan(delta_pco2_gCm3) else 0.0
        dF_TgC_residual = dF_gCs * SEC_PER_YR * TG_PER_G

        huc4_dF_list.append({
            'huc4': huc4, 'huc2': huc2,
            'k_median_ms': k_median,
            'SA_basin_m2': SA_basin,
            'delta_pco2_ppm': delta_pco2_ppm,
            'delta_pco2_gCm3': delta_pco2_gCm3,
            'fitness': fitness,
            'Tw': Tw, 'KH': KH,
            'dF_TgC_yr_residual': dF_TgC_residual,
        })

        #  Standard stats
        stats = {
            'total_flux_gC_yr':  df['reach_flux_gC_yr'].sum(),
            'total_area_m2':     df['reach_area_m2'].sum(),
            'total_length_km':   df['LengthKM'].sum(),
            'mean_FCO2':         df['FCO2_gC_m2_yr'].mean(),
            'mean_pCO2':         df['CO2_ppm'].mean(),
            'mean_width_m':      df['W_m'].mean(),
            'mean_Q_cms':        df['Q_m3_s'].mean(),
            'n_reaches':         n_kept,
            'n_filtered_out':    n_total - n_kept,
            'n_rivers':          (df['waterbody'] == 'River').sum(),
            'n_lakes':           (df['waterbody'] != 'River').sum(),
            'river_flux_gC_yr':  df.loc[df['waterbody'] == 'River', 'reach_flux_gC_yr'].sum(),
            'lake_flux_gC_yr':   df.loc[df['waterbody'] != 'River', 'reach_flux_gC_yr'].sum(),
            'river_area_m2':     df_rivers['reach_area_m2'].sum() if len(df_rivers) else 0,
            'lake_area_m2':      df_lakes['reach_area_m2'].sum()  if len(df_lakes)  else 0,
            'k_river_vals':      df_rivers['k_co2_m_s'].dropna().tolist(),
            'k_lake_vals':       df_lakes['k_co2_m_s'].dropna().tolist(),
        }

        if 'StreamOrde' in df.columns:
            so_flux = df.groupby('StreamOrde')['reach_flux_gC_yr'].sum()
            so_area = df.groupby('StreamOrde')['reach_area_m2'].sum()
            for so in range(1, 12):
                stats[f'flux_SO{so}_gC_yr'] = so_flux.get(so, 0)
                stats[f'area_SO{so}_m2']    = so_area.get(so, 0)

        if huc2 not in huc2_accum:
            huc2_accum[huc2] = []
        huc2_accum[huc2].append(stats)

        del df, df_rivers, df_lakes
        gc.collect()

    huc4_dF = pd.DataFrame(huc4_dF_list)

    #  Aggregate to HUC2
    print("\nAggregating to HUC2...")
    huc2_results = []

    for huc2 in sorted(huc2_accum.keys()):
        huc4_list = huc2_accum[huc2]

        agg = {
            'huc2':              huc2,
            'total_flux_gC_yr':  sum(s['total_flux_gC_yr']  for s in huc4_list),
            'total_area_m2':     sum(s['total_area_m2']     for s in huc4_list),
            'total_length_km':   sum(s['total_length_km']   for s in huc4_list),
            'n_reaches':         sum(s['n_reaches']         for s in huc4_list),
            'n_filtered_out':    sum(s['n_filtered_out']    for s in huc4_list),
            'n_huc4s':           len(huc4_list),
            'river_flux_gC_yr':  sum(s['river_flux_gC_yr']  for s in huc4_list),
            'lake_flux_gC_yr':   sum(s['lake_flux_gC_yr']   for s in huc4_list),
            'river_area_m2':     sum(s['river_area_m2']     for s in huc4_list),
            'lake_area_m2':      sum(s['lake_area_m2']      for s in huc4_list),
        }

        all_k_river = []
        all_k_lake  = []
        for s in huc4_list:
            all_k_river.extend(s['k_river_vals'])
            all_k_lake.extend(s['k_lake_vals'])
        agg['k_median_river'] = np.median(all_k_river) if all_k_river else 0.0
        agg['k_median_lake']  = np.median(all_k_lake)  if all_k_lake  else 0.0

        total_n = agg['n_reaches']
        if total_n > 0:
            def safe_weighted_mean(key):
                pairs = [(s[key], s['n_reaches']) for s in huc4_list
                         if pd.notna(s.get(key))]
                if not pairs:
                    return np.nan
                vals, weights = zip(*pairs)
                return sum(v * w for v, w in zip(vals, weights)) / sum(weights)

            agg['mean_FCO2']    = safe_weighted_mean('mean_FCO2')
            agg['mean_pCO2']    = safe_weighted_mean('mean_pCO2')
            agg['mean_width_m'] = safe_weighted_mean('mean_width_m')
            agg['mean_Q_cms']   = safe_weighted_mean('mean_Q_cms')

        huc2_results.append(agg)

        print(f"  HUC2 {huc2}: {agg['total_flux_gC_yr'] * TG_PER_G:.4f} TgC/yr, "
              f"{agg['total_area_m2'] / 1e6:.0f} km² stream area, "
              f"{agg['n_reaches']:,} reaches ({agg['n_huc4s']} HUC4s)")

    results = pd.DataFrame(huc2_results).set_index('huc2')

    results['Saccardi_flux_TgC_yr'] = results['total_flux_gC_yr'] * TG_PER_G
    results['stream_area_km2']      = results['total_area_m2'] / 1e6
    results['river_flux_TgC_yr']    = results['river_flux_gC_yr'] * TG_PER_G
    results['lake_flux_TgC_yr']     = results['lake_flux_gC_yr']  * TG_PER_G
    results['pct_from_rivers']      = (
        results['river_flux_TgC_yr'] / results['Saccardi_flux_TgC_yr'] * 100
    )

    return results, huc4_dF


# ============================================================
# Saccardi uncertainty — flux-proportional from published ±23 TgC/yr
# ============================================================
def compute_saccardi_uncertainty(results):
    """
    Distribute Saccardi's published ±23 TgC/yr across HUC2s proportionally
    to flux. Each HUC2 gets the same relative uncertainty. Bounds are
    central ± δF (no additional CI scaling).
    """
    total_flux = results['Saccardi_flux_TgC_yr'].sum()
    flux_share = results['Saccardi_flux_TgC_yr'] / total_flux
    dF_per_huc2 = flux_share * SACCARDI_PUBLISHED_DF

    rel_unc = SACCARDI_PUBLISHED_DF / total_flux * 100

    print(f"\n  Saccardi uncertainty (flux-proportional from published ±{SACCARDI_PUBLISHED_DF:.0f} TgC/yr):")
    print(f"  Uniform relative uncertainty: {rel_unc:.1f}%")
    print(f"\n  {'HUC2':>4}  {'flux':>8}  {'share':>7}  {'±δF':>8}")
    for huc2 in results.index:
        f = results.loc[huc2, 'Saccardi_flux_TgC_yr']
        print(f"  {huc2:>4}  {f:>8.3f}  {flux_share[huc2]:>6.1%}  ±{dF_per_huc2[huc2]:>7.3f}")
    print(f"\n  CONUS total: {total_flux:.2f} ± {SACCARDI_PUBLISHED_DF:.1f} TgC/yr")

    return dF_per_huc2


# ============================================================
# Saccardi sensitivity diagnostic — Eqs S16–S17 residuals
# ============================================================
def compute_saccardi_sensitivity_diagnostic(huc4_dF):
    """
    Per-HUC4 δF from calibration residuals (Eqs S16-S17).
    DIAGNOSTIC ONLY — not used for headline bounds.
    """
    dF_huc2 = (huc4_dF.groupby('huc2')
               .agg(
                   dF_TgC_yr_residual=('dF_TgC_yr_residual', 'sum'),
                   n_huc4s=('huc4', 'count'),
                   mean_fitness=('fitness', 'mean'),
                   median_dpco2=('delta_pco2_ppm', 'median'),
                   mean_Tw=('Tw', 'mean'),
                   mean_KH=('KH', 'mean'),
               ))

    computed_total = dF_huc2['dF_TgC_yr_residual'].sum()

    print(f"\n  DIAGNOSTIC: Saccardi δF from Eqs S16-S17 calibration residuals")
    print(f"  Computed CONUS δF:   {computed_total:.2f} TgC/yr")
    print(f"  Published CONUS δF:  {SACCARDI_PUBLISHED_DF:.0f} TgC/yr")
    print(f"  Ratio (computed/published): {computed_total/SACCARDI_PUBLISHED_DF:.2f}")
    print()
    print(f"  {'HUC2':>4}  {'δF_resid':>9}  {'N HUC4':>7}  "
          f"{'mean fit':>10}  {'med δpCO2':>10}  {'Tw °C':>6}  {'KH':>8}")
    for huc2, row in dF_huc2.iterrows():
        print(f"  {huc2:>4}  {row['dF_TgC_yr_residual']:>9.4f}  "
              f"{int(row['n_huc4s']):>7}  "
              f"{row['mean_fitness']:>10.4f}  {row['median_dpco2']:>10.1f}  "
              f"{row['mean_Tw']:>6.1f}  {row['mean_KH']:>8.4f}")

    top5 = huc4_dF.nlargest(5, 'dF_TgC_yr_residual')
    print(f"\n  Top 5 HUC4s by calibration-residual δF:")
    for _, r in top5.iterrows():
        print(f"    {r['huc4']}: δF={r['dF_TgC_yr_residual']:.4f} TgC/yr, "
              f"δpCO2={r['delta_pco2_ppm']:.0f} ppm, "
              f"fitness={r['fitness']:.6f}, "
              f"k_med={r['k_median_ms']:.2e} m/s, "
              f"SA={r['SA_basin_m2']/1e6:.0f} km²")

    return dF_huc2


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 110)
    print("SACCARDI ET AL. (2024) STREAM CO2 EFFLUX BY HUC2")
    print(f"  Source: {SACCARDI_DIR}")
    print(f"  Width filter: W >= {MIN_WIDTH_M} m")
    print(f"  Saccardi uncertainty: published ±{SACCARDI_PUBLISHED_DF:.0f} TgC/yr,"
          f" distributed proportionally to flux")
    print(f"  Butman uncertainty:   ~95% CI rescaled to 90% CI (×1.645/1.960)")
    print(f"  GW uncertainty:       CQR 90% prediction intervals")
    print("=" * 110)

    #  Load climate for KH computation (diagnostic)
    print("\nLoading HUC2 climate for per-basin KH...")
    huc2_Tw = load_huc2_climate(CLIMATE_JSON_PATH)
    for huc2 in sorted(huc2_Tw.keys()):
        Tw = huc2_Tw[huc2]
        KH = compute_KH(Tw)
        print(f"  HUC2 {huc2}: TaTw={Tw:.1f}°C, KH={KH:.4f} mol/(Latm)")

    #  Load HUC4 calibration (diagnostic)
    huc4_calib = load_huc4_calibration(CALIB_PATH, huc2_Tw)

    #  Step 1: Find CSVs
    file_df = discover_csvs(SACCARDI_DIR)

    #  Step 2: Process and aggregate
    results, huc4_dF = process_all_csvs(file_df, huc4_calib)

    #  HUC2 watershed areas
    huc2_areas = load_huc2_areas(HUC_PATH, CONUS_HUC2)
    results['huc2_area_km2'] = huc2_areas
    results['huc2_area_m2']  = huc2_areas * 1e6

    results['Saccardi_yield_gC_m2_yr'] = (
        results['Saccardi_flux_TgC_yr'] / TG_PER_G / results['huc2_area_m2']
    )

    print(f"\n  HUC2 watershed area total: {results['huc2_area_km2'].sum():,.0f} km²")
    print(f"  NHD stream area total:     {results['stream_area_km2'].sum():,.0f} km²")
    print(f"  Saccardi flux total:       {results['Saccardi_flux_TgC_yr'].sum():.2f} TgC/yr")

    #  Saccardi uncertainty (flux-proportional)
    dF_per_huc2 = compute_saccardi_uncertainty(results)
    results['Saccardi_dF_TgC_yr'] = dF_per_huc2
    results['Saccardi_flux_lower_TgC_yr'] = results['Saccardi_flux_TgC_yr'] - dF_per_huc2
    results['Saccardi_flux_upper_TgC_yr'] = results['Saccardi_flux_TgC_yr'] + dF_per_huc2

    #  Sensitivity diagnostic (Eqs S16-S17)
    print("\nComputing sensitivity diagnostic (Eqs S16-S17 residuals)...")
    dF_sens = compute_saccardi_sensitivity_diagnostic(huc4_dF)
    results['Saccardi_dF_residual_diagnostic_TgC_yr'] = dF_sens['dF_TgC_yr_residual']

    #  Join GW flux
    if GW_FLUX_PATH.exists():
        gw = pd.read_csv(GW_FLUX_PATH, index_col='HUC', dtype={'HUC': str})
        results['GW_CO2_flux_TgC_yr'] = gw['GW_CO2_flux_TgC_yr']
        if 'GW_CO2_flux_lower_TgC_yr' in gw.columns:
            results['GW_CO2_flux_lower_TgC_yr'] = gw['GW_CO2_flux_lower_TgC_yr']
            results['GW_CO2_flux_upper_TgC_yr'] = gw['GW_CO2_flux_upper_TgC_yr']
            print("  GW bounds loaded (90% CI)")
        else:
            results['GW_CO2_flux_lower_TgC_yr'] = np.nan
            results['GW_CO2_flux_upper_TgC_yr'] = np.nan

    #  Join Butman (~95%  90% CI)
    if BUTMAN_PATH.exists():
        butman = load_butman(BUTMAN_PATH)
        butman_central = butman['Stream efflux_central']
        butman_lo_95   = butman['Stream efflux_lower']
        butman_hi_95   = butman['Stream efflux_upper']

        results['Butman_stream_central_TgC'] = butman_central

        butman_hw_95 = (butman_hi_95 - butman_lo_95) / 2.0
        butman_hw_90 = butman_hw_95 * (1.645 / 1.960)

        results['Butman_stream_lower_TgC'] = butman_central - butman_hw_90
        results['Butman_stream_upper_TgC'] = butman_central + butman_hw_90

        print(f"\n  Butman bounds rescaled ~95%  90% CI:")
        print(f"    95%: [{butman_lo_95.sum():.1f}, {butman_hi_95.sum():.1f}]")
        print(f"    90%: [{(butman_central - butman_hw_90).sum():.1f}, "
              f"{(butman_central + butman_hw_90).sum():.1f}]")

    #  Derived ratios
    if 'GW_CO2_flux_TgC_yr' in results.columns:
        results['GW_yield_gC_m2_yr'] = (
            results['GW_CO2_flux_TgC_yr'] / TG_PER_G / results['huc2_area_m2']
        )
        results['GW_pct_of_Saccardi'] = (
            results['GW_CO2_flux_TgC_yr'] / results['Saccardi_flux_TgC_yr'] * 100
        )
        results['GW_pct_of_Saccardi_lower'] = (
            results['GW_CO2_flux_lower_TgC_yr'] / results['Saccardi_flux_upper_TgC_yr'] * 100
        )
        results['GW_pct_of_Saccardi_upper'] = (
            results['GW_CO2_flux_upper_TgC_yr'] / results['Saccardi_flux_lower_TgC_yr'] * 100
        )

    if 'Butman_stream_central_TgC' in results.columns:
        results['Butman_yield_gC_m2_yr'] = (
            results['Butman_stream_central_TgC'] / TG_PER_G / results['huc2_area_m2']
        )
        if 'GW_CO2_flux_TgC_yr' in results.columns:
            results['GW_pct_of_Butman'] = (
                results['GW_CO2_flux_TgC_yr'] / results['Butman_stream_central_TgC'] * 100
            )
            results['GW_pct_of_Butman_lower'] = (
                results['GW_CO2_flux_lower_TgC_yr'] / results['Butman_stream_upper_TgC'] * 100
            )
            results['GW_pct_of_Butman_upper'] = (
                results['GW_CO2_flux_upper_TgC_yr'] / results['Butman_stream_lower_TgC'] * 100
            )

    results.index.name = 'HUC'

    #
    # Print results table
    #
    print("\n" + "=" * 180)
    print("RESULTS: Inland Water CO2 Efflux by HUC2")
    print("  Saccardi bounds: published ±23 TgC/yr (flux-proportional)")
    print("  GW bounds: 90% CI (CQR prediction intervals)")
    print("  Butman bounds: 90% CI (rescaled from ~95%)")
    print("=" * 180)

    print(f"{'HUC':>4}  "
          f"{' Saccardi (TgC/yr) ':>34}  "
          f"{' GW flux (TgC/yr) ':>34}  "
          f"{' GW / Saccardi (%) ':>30}  "
          f"{'±δF':>7}  "
          f"{'HUC2 km²':>10}  "
          f"{'Strm km²':>9}")
    print(f"{'':>4}  "
          f"{'lower':>10} {'central':>10} {'upper':>10}  "
          f"{'lower':>10} {'central':>10} {'upper':>10}  "
          f"{'lower':>9} {'central':>9} {'upper':>9}  "
          f"{'(TgC)':>7}  "
          f"{'':>10}  "
          f"{'':>9}")
    print("-" * 180)

    for huc_id, row in results.iterrows():
        print(
            f"{huc_id:>4}  "
            f"{row['Saccardi_flux_lower_TgC_yr']:>10.3f} "
            f"{row['Saccardi_flux_TgC_yr']:>10.3f} "
            f"{row['Saccardi_flux_upper_TgC_yr']:>10.3f}  "
            f"{row.get('GW_CO2_flux_lower_TgC_yr', np.nan):>10.4f} "
            f"{row.get('GW_CO2_flux_TgC_yr', np.nan):>10.4f} "
            f"{row.get('GW_CO2_flux_upper_TgC_yr', np.nan):>10.4f}  "
            f"{row.get('GW_pct_of_Saccardi_lower', np.nan):>8.1f}% "
            f"{row.get('GW_pct_of_Saccardi', np.nan):>8.1f}% "
            f"{row.get('GW_pct_of_Saccardi_upper', np.nan):>8.1f}%  "
            f"{row['Saccardi_dF_TgC_yr']:>7.3f}  "
            f"{row['huc2_area_km2']:>10.0f}  "
            f"{row['stream_area_km2']:>9.0f}"
        )

    #  Totals
    print("-" * 180)
    total_s    = results['Saccardi_flux_TgC_yr'].sum()
    total_s_lo = results['Saccardi_flux_lower_TgC_yr'].sum()
    total_s_hi = results['Saccardi_flux_upper_TgC_yr'].sum()
    total_gw    = results.get('GW_CO2_flux_TgC_yr',       pd.Series(dtype=float)).sum()
    total_gw_lo = results.get('GW_CO2_flux_lower_TgC_yr', pd.Series(dtype=float)).sum()
    total_gw_hi = results.get('GW_CO2_flux_upper_TgC_yr', pd.Series(dtype=float)).sum()

    conus_pct_ce = total_gw    / total_s    * 100 if total_s    > 0 else np.nan
    conus_pct_lo = total_gw_lo / total_s_hi * 100 if total_s_hi > 0 else np.nan
    conus_pct_hi = total_gw_hi / total_s_lo * 100 if total_s_lo > 0 else np.nan

    print(
        f"{'TOT':>4}  "
        f"{total_s_lo:>10.3f} "
        f"{total_s:>10.3f} "
        f"{total_s_hi:>10.3f}  "
        f"{total_gw_lo:>10.4f} "
        f"{total_gw:>10.4f} "
        f"{total_gw_hi:>10.4f}  "
        f"{conus_pct_lo:>8.1f}% "
        f"{conus_pct_ce:>8.1f}% "
        f"{conus_pct_hi:>8.1f}%  "
        f"{SACCARDI_PUBLISHED_DF:>7.1f}  "
        f"{results['huc2_area_km2'].sum():>10.0f}  "
        f"{results['stream_area_km2'].sum():>9.0f}"
    )

    #
    # Summary
    #
    total_b    = results.get('Butman_stream_central_TgC', pd.Series(dtype=float)).sum()
    total_b_lo = results.get('Butman_stream_lower_TgC',  pd.Series(dtype=float)).sum()
    total_b_hi = results.get('Butman_stream_upper_TgC',  pd.Series(dtype=float)).sum()
    total_huc_area = results['huc2_area_m2'].sum()
    total_dF_resid = results.get(
        'Saccardi_dF_residual_diagnostic_TgC_yr', pd.Series(dtype=float)
    ).sum()

    print(f"\n{''*80}")
    print("SUMMARY")
    print(f"{''*80}")
    print(f"  Saccardi: {total_s:.2f} ± {SACCARDI_PUBLISHED_DF:.0f} TgC/yr "
          f"[{total_s_lo:.2f}, {total_s_hi:.2f}]")
    print(f"  Paper:    120 ± 23 TgC/yr (transport model)")
    print(f"  GW:       {total_gw:.4f} TgC/yr "
          f"[{total_gw_lo:.4f}, {total_gw_hi:.4f}] (90% CI)")
    if total_b > 0:
        print(f"  Butman:   {total_b:.1f} TgC/yr "
              f"[{total_b_lo:.1f}, {total_b_hi:.1f}] (90% CI)")
    print(f"\n  DIAGNOSTIC: Eqs S16-S17 calibration-residual δF = "
          f"{total_dF_resid:.2f} TgC/yr "
          f"({total_dF_resid/SACCARDI_PUBLISHED_DF:.1%} of published ±{SACCARDI_PUBLISHED_DF:.0f})")

    #  GW as % of Saccardi
    print(f"\n  GW as % of Saccardi:")
    print(f"    Central:   {conus_pct_ce:.1f}%")
    print(f"    Bounds:    [{conus_pct_lo:.1f}%, {conus_pct_hi:.1f}%]  "
          f"(GW_lo/S_hi, GW_hi/S_lo)")
    pct_lo_gw = total_gw_lo / total_s * 100 if total_s > 0 else np.nan
    pct_hi_gw = total_gw_hi / total_s * 100 if total_s > 0 else np.nan
    print(f"    GW only:   [{pct_lo_gw:.1f}%, {pct_hi_gw:.1f}%]  (Saccardi fixed)")

    #  GW as % of Butman
    if total_b > 0:
        b_pct_ce = total_gw    / total_b    * 100
        b_pct_lo = total_gw_lo / total_b_hi * 100 if total_b_hi > 0 else np.nan
        b_pct_hi = total_gw_hi / total_b_lo * 100 if total_b_lo > 0 else np.nan
        print(f"\n  GW as % of Butman:")
        print(f"    Central:   {b_pct_ce:.1f}%")
        print(f"    Bounds:    [{b_pct_lo:.1f}%, {b_pct_hi:.1f}%]  "
              f"(GW_lo/B_hi, GW_hi/B_lo)")
        pct_lo_gw_b = total_gw_lo / total_b * 100
        pct_hi_gw_b = total_gw_hi / total_b * 100
        print(f"    GW only:   [{pct_lo_gw_b:.1f}%, {pct_hi_gw_b:.1f}%]  (Butman fixed)")

    #  Yields
    print(f"\n  Yields (gC/m²/yr, normalized by HUC2 watershed area):")
    print(f"    Saccardi: {total_s / TG_PER_G / total_huc_area:.4f}")
    if total_b > 0:
        print(f"    Butman:   {total_b / TG_PER_G / total_huc_area:.4f}")
    print(f"    GW:       {total_gw / TG_PER_G / total_huc_area:.4f}")

    #  Save
    results.to_csv(OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")

    huc4_dF_path = OUTPUT_PATH.parent / f'saccardi_huc4_uncertainty_{tag}.csv'
    huc4_dF.to_csv(huc4_dF_path, index=False)
    print(f"Saved HUC4 diagnostic: {huc4_dF_path}")

    return results


if __name__ == '__main__':
    results = main()
