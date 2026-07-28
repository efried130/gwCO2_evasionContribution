"""
data_prep.py

Assemble the groundwater CO2 training table and clean outliers.

Inputs (CSV):
  - co2Point_trainTest_huc12_slim_cat_era5.csv          ERA5 daily point extractions
  - co2Point_trainTest_huc12_slim_cat_era5_monthly.csv  ERA5 monthly point extractions
  - co2Point_trainTest_huc12_slim_cat_gldas.csv         GLDAS point extractions
  - phreeqcOutput.csv                                   PHREEQC-calculated CO2 molality

Output:
  - df_cleaned.csv   analysis-ready training table (one row per well-date)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "data"
ERA5_DAILY   = os.path.join(DATA_DIR, "co2Point_trainTest_huc12_slim_cat_era5.csv")
ERA5_MONTHLY = os.path.join(DATA_DIR, "co2Point_trainTest_huc12_slim_cat_era5_monthly.csv")
GLDAS        = os.path.join(DATA_DIR, "co2Point_trainTest_huc12_slim_cat_gldas.csv")
PHREEQC      = os.path.join(DATA_DIR, "phreeqcOutput.csv")
OUT_CSV      = os.path.join(DATA_DIR, "df_cleaned.csv")



# --- Load and merge point extractions ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import joblib

df_era5 = pd.read_csv(ERA5_DAILY)
print(df_era5.shape, df_era5.columns.values.tolist())
print(df_era5['siteType'].value_counts())

df_era5 = df_era5[
    df_era5['siteType'].str.contains('Well', case=False, na=False)
]
df_era5 = df_era5[~df_era5['siteType'].str.contains('Hyporheic', na=False)]
print(df_era5['siteType'].value_counts())

df_era5 = df_era5.drop(columns=['AvgSurfT_inst_',
 'ESoil_tavg_','ET_water_mean_','Ec_mean_','Ei_mean_','Es_mean_',
 'Evap_tavg_','GPP_mean_','PotEvap_tavg_','Qair_f_inst_','Qs_acc_','Qsb_acc_',
 'Qsm_acc_','Rainf_f_tavg_','Rainf_tavg_','RootMoist_inst_','SWE_inst_','SnowDepth_inst_',
 'SoilMoi0_10cm_inst_','SoilMoi100_200cm_inst_','SoilTMP0_10cm_inst_',
 'SoilTMP100_200_inst_','Tair_f_inst_','Tveg_tavg_',])
print("df_era5 sample:", df_era5['Date'].head(10).tolist())

era5_cols_month = ['d2m', 't2m', 'stl1', 'stl2', 'stl3', 'stl4', 'snowc', 'rsn', 'sde', 'sd', 'sf',
             'smlt', 'tsn', 'src', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'fal', 'slhf', 'ssr',
             'str', 'sshf', 'ssrd', 'strd', 'evabs', 'evaow', 'evatc', 'evavt', 'pev', 'ro',
             'es', 'ssro', 'sro', 'e', 'sp', 'tp', 'lai_hv', 'lai_lv']

df_era5_month = pd.read_csv(ERA5_MONTHLY)
rename_map = {col: f'{col}_month' for col in era5_cols_month if col in df_era5_month.columns}
df_era5_month = df_era5_month.rename(columns=rename_map)
print(df_era5_month.shape, df_era5_month.columns.values.tolist())
print("df_era5 month sample:", df_era5_month['Date'].head(10).tolist())

df_era5_gldas = pd.read_csv(GLDAS)
df_era5_gldas = df_era5_gldas[['PFAF_ID', 'HYBAS_ID', 'MonitoringLocation', 'Date', 'CO2_ppm', 'AvgSurfT_inst', 'ESoil_tavg', 'Evap_tavg', 'PotEvap_tavg', 'Qair_f_inst', 'Qs_acc', 'Qsb_acc', 'Qsm_acc',
                               'Rainf_f_tavg', 'Rainf_tavg', 'RootMoist_inst', 'SWE_inst', 'SnowDepth_inst', 'SoilMoi0_10cm_inst',
                               'SoilMoi100_200cm_inst', 'SoilTMP0_10cm_inst', 'SoilTMP100_200_inst', 'Tair_f_inst', 'Tveg_tavg']]
print(df_era5_gldas.shape, df_era5_gldas.columns.values.tolist())
print("df_gldas sample:", df_era5_gldas['Date'].head(10).tolist())

df1 = df_era5.merge(df_era5_month, how = 'left', on = ['CO2_ppm', 'Date', 'MonitoringLocation', 'PFAF_ID', 'HYBAS_ID'])
df = df1.merge(df_era5_gldas, how = 'left', on = ['CO2_ppm', 'Date', 'MonitoringLocation', 'PFAF_ID', 'HYBAS_ID'])

#'geometry',
df = df.dropna(subset = 't2m')
print(df.shape)

print("CO2_ppm column summary:")
print(df['CO2_ppm'].describe())

print(f"\nCO2_ppm data points: {df['CO2_ppm'].count()}")
print(f"Missing values: {df['CO2_ppm'].isnull().sum()}")

# --- Merge PHREEQC CO2 and convert molality to mol/L ---
df_phreeqc = pd.read_csv(PHREEQC)

# Parse PHREEQC dates with explicit format, localize to UTC
df_phreeqc['Date'] = pd.to_datetime(df_phreeqc['Date'], format='%m/%d/%y', errors='coerce', utc=True)
df_phreeqc['Date'] = df_phreeqc['Date'].dt.normalize()
df_phreeqc['Date'] = df_phreeqc['Date'].dt.strftime('%Y-%m-%d')

# Parse the main df dates, localize to UTC
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
df['Date'] = df['Date'].dt.normalize()
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

print("PHREEQC dates sample:", df_phreeqc['Date'].head(10).tolist())
print("df sample:", df['Date'].head(10).tolist())

print(df_phreeqc.shape, df_phreeqc.columns.values.tolist())


# merge
df = df.merge(
    df_phreeqc[['Date', 'MonitoringLocation','molality_CO2']],
    how='left',
    on=['Date', 'MonitoringLocation']
)
df = df.dropna(subset='molality_CO2')
print(f"\nmolality_CO2 data points: {df['molality_CO2'].count()}")
print(df.shape)

# Convert molality to mol/L
Mw_CO2 = 0.04401
rho = 1.0
df['CO2_aq'] = (df['molality_CO2'] * rho) / (1 + df['molality_CO2'] * Mw_CO2)

print(df.shape, df.columns.values.tolist())


# --- Outlier cleaning ---

def filter_within_well_outliers(df, target='CO2_aq', well_col='MonitoringLocation',
                                 max_log_deviation=2.0, min_n=4):
    """
    Remove observations where log10(CO2) deviates more than
    max_log_deviation from that well's median.
    Only applied to wells with >= min_n observations.
    """
    df = df.copy()
    df['log_co2'] = np.log10(df[target])

    well_stats = df.groupby(well_col)['log_co2'].agg(['median','count','std'])
    well_stats.columns = ['well_log_median','well_n','well_log_std']
    df = df.join(well_stats, on=well_col)

    # For wells with enough obs, flag extreme deviations
    has_enough = df['well_n'] >= min_n
    deviation  = np.abs(df['log_co2'] - df['well_log_median'])

    keep = ~has_enough | (deviation <= max_log_deviation)
    n_removed = (~keep).sum()
    print(f"Within-well filter: removed {n_removed:,} ({n_removed/len(df):.1%})")
    return df[keep].drop(columns=['log_co2','well_log_median','well_n','well_log_std'])

from sklearn.neighbors import NearestNeighbors

def filter_spatial_isolates(df, feature_cols, n_neighbors=5,
                             isolation_percentile=99):
    """
    Remove points whose mean distance to k nearest neighbors
    exceeds the isolation_percentile threshold.
    These are covariate-space outliers — either unusual environments
    or mismeasured locations.
    """
    X = df[feature_cols].dropna().values
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    mean_dist = distances[:, 1:].mean(axis=1)  # exclude self

    threshold = np.percentile(mean_dist, isolation_percentile)
    keep_mask = mean_dist <= threshold
    n_removed = (~keep_mask).sum()
    print(f"Isolation filter: removed {n_removed:,} ({n_removed/len(df[feature_cols].dropna()):.1%})")
    return df[df[feature_cols].notna().all(axis=1)][keep_mask]

def stratified_outlier_filter(df, target='CO2_aq',
                               strata_cols=['season', 'HUC2'],
                               iqr_threshold=3.5):
    """
    IQR outlier filter applied within each stratum (season × HUC2).
    Much more conservative than global IQR — only removes values
    that are extreme *within their own environmental context*.
    """
    df = df.copy()
    df['log_co2'] = np.log10(df[target].clip(lower=1e-12))
    flag = pd.Series(False, index=df.index)

    for strat_vals, group in df.groupby(strata_cols):
        if len(group) < 20:
            continue
        q25, q75 = group['log_co2'].quantile([0.25, 0.75])
        iqr = q75 - q25
        if iqr == 0:
            continue
        lo = q25 - iqr_threshold * iqr
        hi = q75 + iqr_threshold * iqr
        flag.loc[group.index] = (
            (group['log_co2'] < lo) | (group['log_co2'] > hi)
        )

    n_removed = flag.sum()
    print(f"Stratified IQR: removed {n_removed:,} ({n_removed/len(df):.1%})")
    return df[~flag].drop(columns=['log_co2'])

# 1. Hard physical limits (always first)
df = df[(df['CO2_aq'] >= 1e-6) & (df['CO2_aq'] <= 0.05)]
df = df[(df['ResultMeasure_pH'] >= 4.0) & (df['ResultMeasure_pH'] <= 10.5)]
df = df[(df['Alkalinity_mg.L.CaCO3'] >= 0) & (df['Alkalinity_mg.L.CaCO3'] <= 2000)]
df = df[(df['ResultMeasure_tempC'] >= 0) & (df['ResultMeasure_tempC'] <= 50)]


# 3. Within-well temporal consistency
df = filter_within_well_outliers(df, max_log_deviation=2.0, min_n=4)

# 4. Stratified IQR (replaces your current global IQR in CONFIG)
df = stratified_outlier_filter(df, iqr_threshold=3.5)
NUMERIC_FEATURES = [
    # ERA5 month
# 'd2m_month','t2m_month','stl1_month',
#  'stl2_month','stl3_month','stl4_month','snowc_month',
#  'rsn_month','sde_month','sd_month','sf_month',
#  'smlt_month','tsn_month', 'src_month','swvl1_month',
#  'swvl2_month','swvl3_month','swvl4_month','fal_month',
#  'slhf_month','ssr_month','str_month','sshf_month',
#  'ssrd_month','strd_month','evabs_month','evaow_month',
#  'evatc_month','evavt_month','pev_month','ro_month',
#  'es_month','ssro_month','sro_month','e_month',
#  'sp_month','tp_month','lai_hv_month','lai_lv_month',
    # ERA5 core
 't2m','sp','d2m','fal','ssrd','strd','e',
 'tp','sshf','slhf','ssr',
 'evatc','evabs','evavt','pev',
 'stl1','stl2','stl3','stl4',
 'swvl1','swvl2','swvl3','swvl4',
    # HydroATLAS static
 'aet_mm_s',
 'cmi_ix_s',
 'pet_mm_s',
 'pre_mm_s',
 'snw_pc_s',
 'swc_pc_s',
 'tmp_dc_s',
 'COAST',
 'DIST_MAIN',
 'DIST_SINK',
 'ORDER_',
 'SUB_AREA',
 'UP_AREA',
 'ari_ix_sav',
 # 'cls_cl_smj',
 'cly_pc_sav',
 # 'clz_cl_smj',
 'crp_pc_sse',
 'dor_pc_pva',
 'dtb',
 'ele_mt_sav',
 'ero_kh_sav',
 # 'fec_cl_smj',
 # 'fmh_cl_smj',
 'for_pc_sse',
 'gad_id_smj',
 'gdp_ud_sav',
 'gdp_ud_ssu',
 'gla_pc_sse',
 # 'glc_cl_smj',
 'gwt_cm_sav',
 'hdi_ix_sav',
 'hft_ix_s09',
 'hft_ix_s93',
 'inu_pc_slt',
 'ire_pc_sse',
 'kar_pc_sse',
 # 'lit_cl_smj',
 'lka_pc_sse',
 'nli_ix_sav',
 'pac_pc_sse',
 # 'pnv_cl_smj',
 'pop_ct_ssu',
 'ppd_pk_sav',
 'prm_pc_sse',
 'pst_pc_sse',
 'rdd_mk_sav',
 'ria_ha_ssu',
 'riv_tc_ssu',
 'sgr_dk_sav',
 'slp_dg_sav',
 'slt_pc_sav',
 'snd_pc_sav',
 'soc_th_sav',
 # 'tbi_cl_smj',
 # 'tec_cl_smj',
 'urb_pc_sse',
    # 'swc_pc_s', 'ari_ix_sav', 'dtb', 'DIST_MAIN',
    # 'ero_kh_sav', 'ele_mt_sav', 'soc_th_sav', 'Dd', 'lka_pc_sse',
    # Temporal
    'season', 'year',
    # GLDAS monthly
 #    'AvgSurfT_inst','ESoil_tavg','Evap_tavg',
 # 'PotEvap_tavg','Qair_f_inst','Rainf_f_tavg',
 # 'Rainf_tavg','RootMoist_inst','SWE_inst',
 # 'SnowDepth_inst','SoilMoi0_10cm_inst','SoilMoi100_200cm_inst',
 # 'SoilTMP0_10cm_inst','SoilTMP100_200_inst','Tair_f_inst','Tveg_tavg',
]
# 5. Spatial isolation (optional, only if generalizing to ungauged areas)
df = filter_spatial_isolates(df, feature_cols=NUMERIC_FEATURES[:20])

# --- Remove duplicate ERA5 combinations ---
# Define ERA5 columns to check for duplicates
era5_cols = ['t2m', 'sp', 'd2m', 'fal', 'ssrd', 'strd', 'ro', 'sro', 'e', 'tp',
             'ssro', 'sshf', 'slhf', 'ssr', 'evatc', 'evabs', 'evavt', 'pev',
             'stl1', 'stl2', 'stl3', 'stl4', 'swvl1', 'swvl2', 'swvl3', 'swvl4']

print(f"Original dataset shape: {df.shape}")
print(f"Original CO2 measurements: {df['CO2_ppm'].count()}")

# Find rows with identical ERA5 values but different CO2 measurements
# Group by ERA5 columns and check if multiple CO2 values exist for same ERA5 data
grouped = df.groupby(era5_cols)['CO2_ppm'].agg(['count', 'nunique'])
duplicates_mask = grouped['count'] > grouped['nunique']
duplicate_groups = grouped[duplicates_mask]

print(f"\nNumber of ERA5 combinations with multiple different CO2 values: {len(duplicate_groups)}")
print(f"Total rows affected: {duplicate_groups['count'].sum()}")

# Keep only one row per unique combination of ERA5 values
# This keeps the first occurrence of each ERA5 combination
df_cleaned = df.drop_duplicates(subset=era5_cols, keep='first')

print(f"\nCleaned dataset shape: {df_cleaned.shape}")
print(f"Cleaned CO2 measurements: {df_cleaned['CO2_ppm'].count()}")
print(f"Rows removed: {len(df) - len(df_cleaned)}")
print(f"Unique PFAF_IDs after cleaning: {df_cleaned['PFAF_ID'].nunique()}")

# Show some examples of what was removed
if len(duplicate_groups) > 0:
    print("\nExample of duplicate ERA5 data with different CO2 values:")
    # Get first duplicate group
    first_dup_vals = duplicate_groups.index[0]
    first_dup_vals = duplicate_groups.index[1]
    first_dup_vals = duplicate_groups.index[10]

    # Find all rows matching this ERA5 combination
    mask = True
    for col, val in zip(era5_cols, first_dup_vals):
        mask = mask & (df[col] == val)
    example_dups = df[mask][['Date', 'MonitoringLocation', 'CO2_ppm', 't2m', 'sp', 'tp']]
    print(example_dups.head(10))

# --- Drop known bad wells ---
# Remove Long Island

df_cleaned = df_cleaned[~df_cleaned['MonitoringLocation'].str.startswith('USGS-404')]
df_cleaned = df_cleaned[~df_cleaned['MonitoringLocation'].str.startswith('USGS-390')]
catastrophic_wells = [
    'USGS-393633103512300',
    'USGS-393702103544100',
    'USGS-394333103525100',
]
df_cleaned = df_cleaned[~df_cleaned['MonitoringLocation'].isin(catastrophic_wells)]

# Count unique PFAF_IDs

original_pfaf_count = df_cleaned['PFAF_ID'].nunique()
print(f"{original_pfaf_count:,} unique PFAF_IDs")


# --- Derive HUC2 ---
df_cleaned['HUC8'] = df_cleaned['HUC8'].astype(str)
print(df_cleaned.HUC8.tail())
df_cleaned['HUC2'] = df_cleaned['HUC8'].str.zfill(8)#[:2]
df_cleaned['HUC2'] = df_cleaned['HUC2'].str[:2]
print(df_cleaned.HUC2.tail())
# df_cleaned['HUC2'].value_counts()

# --- Save ---
df_cleaned.to_csv(OUT_CSV, index=False)
