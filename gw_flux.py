"""
gw_flux.py

Groundwater CO2 flux per HUC2: gridded CO2(aq) x ERA5-Land subsurface runoff x
cell area, with split-conformal prediction intervals propagated through the flux.
Writes gw_co2_flux_by_huc2_*.csv used by the Saccardi/Liu contribution scripts.
"""
co2_predictions = "co2_predictions_aq_uncertainty2"

"""
gw_co2_flux.py — Calculate GW CO2 flux per HUC2 WITH UNCERTAINTY.

Propagates split conformal prediction intervals through the flux calculation:
    flux = CO2 × runoff × cell_area × 1000 × MW_C

MONTHLY vs ANNUAL (USE_MONTHLY)
  USE_MONTHLY = True  (default):
    Flux is built as Σ_m (CO2_{cell,month} × ff_{cell,month}) where ff_{cell,month}
    uses the ERA5-Land MONTHLY ssro climatology. This captures seasonal covariance
    between runoff and predicted [CO2] (e.g. snowmelt-season runoff meeting different
    CO2 than late-summer baseflow). Because Σ_m runoff_m = runoff_yr, flattening the
    monthly factors to runoff_yr/12 reproduces the annual result exactly — that is the
    regression test.

  USE_MONTHLY = False:
    Original behaviour — annual-mean CO2 × annual flux_factor. Use to reproduce the
    7.96 TgC/yr / 12.9% baseline.

UNCERTAINTY AGGREGATION (HUC4 block):
  Primary method: linear sum of cell σ within each HUC4, then quadrature
  across all HUC4s for the CONUS total. In the monthly path, per-cell σ is the
  linear sum of monthly σ (months treated as perfectly correlated within a cell,
  consistent with the linear-within-block philosophy).

  Two per-HUC2 σ columns are written to CSV for downstream use:
    GW_CO2_flux_sigma_huc4_TgC_yr — PRIMARY: quadrature of linear-within-HUC4 σ
    GW_CO2_flux_sigma_huc2_TgC_yr — REFERENCE: linear-within-HUC2 σ (pessimistic)

  NOTE: lower/upper CI bounds are NOT written to CSV — downstream computes
  them from σ using quadrature across HUC4s (or HUC2s for sensitivity).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
IS_PPM = not co2_predictions.endswith('_aq_uncertainty2')

# Toggle: True = monthly covariance-aware flux; False = original annual calc.
USE_MONTHLY = True

BASE_DIR  = Path("data")
CO2_DIR   = BASE_DIR / co2_predictions
HUC2_PATH = BASE_DIR / 'shp_files/Watershed_Boundary_Dataset_HUC_2.gpkg'
HUC4_PATH = BASE_DIR / 'shp_files/wbd_HU04/HU04.shp'

tag = 'ppm' if IS_PPM else 'aq'
CI_LEVEL = 90

print(f"CO2 source : {co2_predictions}")
print(f"Input units: {'ppm  Henrys law' if IS_PPM else 'mol/L (aqueous, used directly)'}")
print(f"CI level:    {CI_LEVEL}%")
print(f"Mode:        {'MONTHLY (seasonal covariance)' if USE_MONTHLY else 'ANNUAL (baseline)'}")

# --- Time period configs ---
# monthly_runoff_path: 12-month ssro climatology (m/month). Only the 1979-2014
# climatology exists; both configs point at it. For full period-matching, build a
# 1990-2010 monthly climatology and repoint 'butman' here — the seasonal SHAPE
# barely differs between the two windows, so this is a minor consistency nicety.
MONTHLY_CLIM = BASE_DIR / 'runoff/era5_land_monthly_ssro_climatology_1979_2014.nc'
MONTHLY_CLIM_BUTMAN = BASE_DIR / 'runoff/era5_land_monthly_ssro_climatology_1990_2010.nc'

CONFIGS = {
    'saccardi_liu': {
        'runoff_path':         BASE_DIR / 'runoff/era5_land_mean_annual_runoff_1979_2014_nofreeze.nc',
        'monthly_runoff_path': MONTHLY_CLIM,
        'year_range':          range(1979, 2015),
        'output_path':         BASE_DIR / f'gw_co2_flux_by_huc2_1979_2014_{tag}.csv',
        'cell_output_path':    BASE_DIR / f'gw_co2_flux_gridcells_1979_2014_{tag}.csv',
    },
    'butman': {
        'runoff_path':         BASE_DIR / 'runoff/era5_land_mean_annual_runoff_1990_2010.nc',
        'monthly_runoff_path': MONTHLY_CLIM_BUTMAN,
        'year_range':          range(1990, 2011),
        'output_path':         BASE_DIR / f'gw_co2_flux_by_huc2_1990_2010_{tag}.csv',
        'cell_output_path':    BASE_DIR / f'gw_co2_flux_gridcells_1990_2010_{tag}.csv',
    },
}

CONUS_HUC2 = [str(i).zfill(2) for i in range(1, 19)]
MONTH_COLS = [f'ff_m{m}' for m in range(1, 13)]

TEMP_C    = 15.0
KH_25C    = 0.034
DH_OVER_R = 2400.0
T_REF     = 298.15
T_ACT     = TEMP_C + 273.15
KH_CO2    = KH_25C * np.exp(DH_OVER_R * (1.0 / T_ACT - 1.0 / T_REF))

MW_C      = 12.011
MW_CO2    = 44.009
TG_PER_G  = 1e-12
Z_90      = 1.645


def cell_area_m2(lats, dlon=0.1, dlat=0.1):
    R = 6_371_000.0
    lat1 = np.radians(lats - dlat / 2.0)
    lat2 = np.radians(lats + dlat / 2.0)
    return R**2 * np.abs(np.sin(lat2) - np.sin(lat1)) * np.radians(dlon)


# ============================================================
# MONTHLY RUNOFF INTERPOLATORS
# ============================================================
def _load_monthly_runoff_interps(monthly_path):
    """Return a list of 12 nearest-neighbour interpolators (m/month) from the
    monthly ssro climatology, with coords made ascending for RegularGridInterpolator."""
    ds = xr.open_dataset(monthly_path)
    var = 'ssro' if 'ssro' in ds.data_vars else list(ds.data_vars)[0]
    mlat = ds['latitude'].values.astype(np.float64)
    mlon = ds['longitude'].values.astype(np.float64)
    data = np.asarray(ds[var].values, dtype=np.float64)   # (12, lat, lon)
    ds.close()
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)

    if mlat[0] > mlat[-1]:
        mlat = mlat[::-1]
        data = data[:, ::-1, :]
    if mlon.max() > 180:
        shift = mlon >= 180
        mlon[shift] -= 360
        order = np.argsort(mlon)
        mlon = mlon[order]
        data = data[:, :, order]

    interps = []
    for m in range(12):
        interps.append(RegularGridInterpolator(
            (mlat, mlon), data[m],
            method='nearest', bounds_error=False, fill_value=np.nan))
    return interps


# ============================================================
# BUILD CELL LOOKUP
# ============================================================
def build_cell_lookup(co2_dir, runoff_path, huc2_path, huc4_path, year_range,
                      monthly_runoff_path=None, use_monthly=False):
    print("Building cell lookup table...")
    sample_year = None
    for year in year_range:
        year_dir = co2_dir / f'year={year}'
        if year_dir.exists():
            sample_year = year
            break

    coords = []
    for month in range(1, 13):
        month_dir = co2_dir / f'year={sample_year}' / f'month={month}'
        if not month_dir.exists():
            continue
        for pq_file in month_dir.glob('*.parquet'):
            df = pd.read_parquet(pq_file, columns=['latitude', 'longitude'])
            df['lat'] = df['latitude'].round(1)
            df['lon'] = df['longitude'].round(1)
            coords.append(df[['lat', 'lon']].drop_duplicates())

    cells = pd.concat(coords).drop_duplicates().reset_index(drop=True)
    print(f"  {len(cells)} unique grid cells from year {sample_year}")

    cells['cell_area_m2'] = cell_area_m2(cells['lat'].values)

    #  Annual runoff (always loaded: used for USE_MONTHLY=False and as reference)
    print("  Loading ERA5 annual runoff...")
    ds = nc.Dataset(runoff_path, 'r')
    era5_lat = ds.variables['latitude'][:].astype(np.float64)
    era5_lon = ds.variables['longitude'][:].astype(np.float64)
    var_name = 'ssro' if 'ssro' in ds.variables else 'ro'
    runoff_data = ds.variables[var_name][:].astype(np.float64)
    ds.close()

    if isinstance(runoff_data, np.ma.MaskedArray):
        runoff_data = runoff_data.filled(np.nan)
    if era5_lat[0] > era5_lat[-1]:
        era5_lat = era5_lat[::-1]
        runoff_data = runoff_data[::-1, :]
    if era5_lon.max() > 180:
        shift = era5_lon >= 180
        era5_lon[shift] -= 360
        sort_idx = np.argsort(era5_lon)
        era5_lon = era5_lon[sort_idx]
        runoff_data = runoff_data[:, sort_idx]

    interp = RegularGridInterpolator(
        (era5_lat, era5_lon), runoff_data,
        method='nearest', bounds_error=False, fill_value=np.nan
    )
    cells['runoff_annual_m_yr'] = np.clip(interp(cells[['lat', 'lon']].values), 0, None)

    #  Monthly runoff  per-month flux factors
    if use_monthly and monthly_runoff_path is not None and Path(monthly_runoff_path).exists():
        print(f"  Loading MONTHLY ssro climatology: {Path(monthly_runoff_path).name}")
        interps_m = _load_monthly_runoff_interps(monthly_runoff_path)
        pts = cells[['lat', 'lon']].values

        r_months = np.zeros((12, len(cells)), dtype=np.float64)
        for m in range(12):
            r_months[m] = np.clip(interps_m[m](pts), 0, None)

        # annual runoff for this path = sum of monthly (self-consistent with the flux)
        cells['runoff_m_yr'] = np.nansum(r_months, axis=0)

        # per-month flux factor (same structure as annual flux_factor)
        wv_months = r_months * cells['cell_area_m2'].values[None, :] * 1000.0  # L/yr per month
        if IS_PPM:
            ff_months = KH_CO2 * 1e-6 * wv_months * MW_C
        else:
            ff_months = wv_months * MW_C
        for m in range(12):
            cells[MONTH_COLS[m]] = np.nan_to_num(ff_months[m], nan=0.0)

        # report a monthly/annual consistency hint
        ann = cells['runoff_annual_m_yr']
        mon = cells['runoff_m_yr']
        both = (ann > 0) & (mon > 0)
        if both.any():
            ratio = float(mon[both].mean() / ann[both].mean())
            print(f"  Monthly-sum vs annual-file runoff (mean ratio): {ratio:.3f} "
                  f"(grids/years differ slightly; ~0.95–1.05 expected)")
    else:
        if use_monthly:
            print("   USE_MONTHLY requested but monthly file missing — falling back to annual.")
        cells['runoff_m_yr'] = cells['runoff_annual_m_yr']

    valid_ro = cells.loc[cells['runoff_m_yr'] > 0, 'runoff_m_yr']
    print(f"  Runoff: {valid_ro.count()} cells with >0, "
          f"median={valid_ro.median():.4f}, max={valid_ro.max():.4f} m/yr")

    print("  Assigning HUC2 regions...")
    hucs2 = gpd.read_file(huc2_path)
    hucs2 = hucs2[hucs2['huc2'].isin(CONUS_HUC2)].to_crs('EPSG:4326')

    geom = gpd.points_from_xy(cells['lon'], cells['lat'])
    cells_gdf = gpd.GeoDataFrame(cells, geometry=geom, crs='EPSG:4326')
    joined2 = gpd.sjoin(cells_gdf, hucs2[['huc2', 'geometry']], how='left', predicate='within')
    joined2 = joined2.drop_duplicates(subset=['lat', 'lon'], keep='first')
    cells['huc2'] = joined2['huc2'].values
    n_assigned = cells['huc2'].notna().sum()
    print(f"  {n_assigned}/{len(cells)} cells assigned to HUC2 ({100*n_assigned/len(cells):.1f}%)")

    print("  Assigning HUC4 regions...")
    hucs4 = gpd.read_file(huc4_path).to_crs('EPSG:4326')
    if 'huc4' not in hucs4.columns:
        for col in ['HUC4', 'huc4', 'HUC_4', 'HYBAS_ID']:
            if col in hucs4.columns:
                hucs4 = hucs4.rename(columns={col: 'huc4'})
                break
    hucs4['huc2_prefix'] = hucs4['huc4'].str[:2]
    hucs4 = hucs4[hucs4['huc2_prefix'].isin(CONUS_HUC2)]
    joined4 = gpd.sjoin(cells_gdf, hucs4[['huc4', 'geometry']], how='left', predicate='within')
    joined4 = joined4.drop_duplicates(subset=['lat', 'lon'], keep='first')
    cells['huc4'] = joined4['huc4'].values
    n_assigned4 = cells['huc4'].notna().sum()
    print(f"  {n_assigned4}/{len(cells)} cells assigned to HUC4 ({100*n_assigned4/len(cells):.1f}%)")

    # Annual flux_factor (drives flux only when USE_MONTHLY=False; also stored in cell CSV)
    water_volume_factor = cells['runoff_m_yr'] * cells['cell_area_m2'] * 1000.0
    if IS_PPM:
        cells['flux_factor'] = KH_CO2 * 1e-6 * water_volume_factor * MW_C
        print(f"  flux_factor includes Henry's law (KH={KH_CO2:.5f} at {TEMP_C}°C)")
    else:
        cells['flux_factor'] = water_volume_factor * MW_C
        print(f"  flux_factor uses aqueous [CO2] directly (no Henry's law)")

    lookup = cells[cells['huc2'].notna() & cells['huc4'].notna() & (cells['runoff_m_yr'] > 0)].copy()
    lookup = lookup.set_index(['lat', 'lon'])
    print(f"  Final lookup: {len(lookup)} valid cells (huc2 + huc4 assigned, runoff > 0)")

    # Regression guard: monthly factors should sum to ~the annual flux_factor
    if use_monthly and MONTH_COLS[0] in lookup.columns:
        ff_sum = lookup[MONTH_COLS].sum(axis=1)
        rel = float((ff_sum.sum() / lookup['flux_factor'].sum()))
        print(f"  [check] Σ monthly ff / annual flux_factor = {rel:.3f} "
              f"(≈1.0 expected; ~20× means days-in-month fix missing)")

    return lookup


# ============================================================
# YEAR AGGREGATION (shared by both paths)
# ============================================================
def _aggregate_year(merged, has_ci):
    """merged: per-cell DataFrame with columns huc2, huc4, flux_TgC,
    sigma_flux_TgC (if has_ci), CO2_predicted. Returns the 4 per-HUC2 Series."""
    huc_flux = merged.groupby('huc2')['flux_TgC'].sum()
    huc_co2  = merged.groupby('huc2')['CO2_predicted'].mean()

    if has_ci:
        # PRIMARY: linear within HUC4, quadrature across HUC4s within HUC2
        huc4_sigma = merged.groupby('huc4')['sigma_flux_TgC'].sum()
        huc4_df = huc4_sigma.reset_index()
        huc4_df['huc2'] = huc4_df['huc4'].str[:2]
        huc2_sigma_huc4 = huc4_df.groupby('huc2')['sigma_flux_TgC'].apply(
            lambda x: np.sqrt((x**2).sum()))
        # REFERENCE: linear within HUC2
        huc2_sigma_huc2 = merged.groupby('huc2')['sigma_flux_TgC'].sum()
    else:
        huc2_sigma_huc4 = None
        huc2_sigma_huc2 = None

    return huc_flux, huc_co2, huc2_sigma_huc4, huc2_sigma_huc2


# ============================================================
# PER-YEAR MERGED BUILDERS
# ============================================================
def _merged_annual(year_dir, lookup, has_ci, lo_col, hi_col):
    """Original behaviour: annual-mean CO2 × annual flux_factor."""
    monthly_dfs = []
    for month in range(1, 13):
        month_dir = year_dir / f'month={month}'
        if not month_dir.exists():
            continue
        for pq_file in month_dir.glob('*.parquet'):
            read_cols = ['latitude', 'longitude', 'CO2_predicted']
            if has_ci:
                read_cols += [lo_col, hi_col]
            df = pd.read_parquet(pq_file, columns=read_cols)
            df['lat'] = df['latitude'].round(1)
            df['lon'] = df['longitude'].round(1)
            keep = ['lat', 'lon', 'CO2_predicted'] + ([lo_col, hi_col] if has_ci else [])
            monthly_dfs.append(df[keep])
    if not monthly_dfs:
        return None

    year_data = pd.concat(monthly_dfs, ignore_index=True)
    del monthly_dfs; gc.collect()

    agg_cols = {'CO2_predicted': 'mean'}
    if has_ci:
        agg_cols[lo_col] = 'mean'
        agg_cols[hi_col] = 'mean'
    year_mean = year_data.groupby(['lat', 'lon']).agg(agg_cols)
    del year_data; gc.collect()

    merged = lookup[['huc2', 'huc4', 'flux_factor']].join(year_mean, how='inner')
    merged['flux_TgC'] = merged['CO2_predicted'] * merged['flux_factor'] * TG_PER_G
    if has_ci:
        merged['sigma_co2'] = (merged[hi_col] - merged[lo_col]) / 2.0
        merged['sigma_flux_TgC'] = merged['sigma_co2'] * merged['flux_factor'] * TG_PER_G
    return merged


def _merged_monthly(year_dir, lookup, has_ci, lo_col, hi_col):
    """Monthly path: Σ_m (CO2_m × ff_m), σ summed linearly across months per cell."""
    idx = lookup.index
    flux_cell  = pd.Series(0.0, index=idx)
    sigma_cell = pd.Series(0.0, index=idx)
    co2_sum    = pd.Series(0.0, index=idx)
    months_done = 0

    for month in range(1, 13):
        month_dir = year_dir / f'month={month}'
        if not month_dir.exists():
            continue
        dfs = []
        for pq_file in month_dir.glob('*.parquet'):
            read_cols = ['latitude', 'longitude', 'CO2_predicted']
            if has_ci:
                read_cols += [lo_col, hi_col]
            df = pd.read_parquet(pq_file, columns=read_cols)
            df['lat'] = df['latitude'].round(1)
            df['lon'] = df['longitude'].round(1)
            dfs.append(df)
        if not dfs:
            continue

        md = pd.concat(dfs, ignore_index=True)
        agg = {'CO2_predicted': 'mean'}
        if has_ci:
            agg[lo_col] = 'mean'
            agg[hi_col] = 'mean'
        mm = md.groupby(['lat', 'lon']).agg(agg)
        del md, dfs; gc.collect()

        ff_m  = lookup[MONTH_COLS[month - 1]]                       # per-cell, this month
        co2_m = mm['CO2_predicted'].reindex(idx).fillna(0.0)
        flux_cell += co2_m * ff_m * TG_PER_G
        if has_ci:
            sig_m = ((mm[hi_col] - mm[lo_col]) / 2.0).reindex(idx).fillna(0.0)
            sigma_cell += sig_m * ff_m * TG_PER_G                   # linear across months (correlated)
        co2_sum += co2_m
        months_done += 1
        del mm; gc.collect()

    if months_done == 0:
        return None

    merged = pd.DataFrame({
        'huc2':          lookup['huc2'].values,
        'huc4':          lookup['huc4'].values,
        'flux_TgC':      flux_cell.values,
        'CO2_predicted': (co2_sum / months_done).values,   # diagnostic only
    }, index=idx)
    if has_ci:
        merged['sigma_flux_TgC'] = sigma_cell.values
    return merged


# ============================================================
# STREAMING FLUX COMPUTATION — with uncertainty
# ============================================================
def compute_flux_streaming(co2_dir, year_range, lookup, ci_level=90, use_monthly=False):
    lo_col = f'CI_lower_{ci_level}'
    hi_col = f'CI_upper_{ci_level}'

    # Detect CI columns from a sample parquet
    has_ci = False
    for year in year_range:
        for month in range(1, 13):
            md = co2_dir / f'year={year}' / f'month={month}'
            if not md.exists():
                continue
            sample_files = list(md.glob('*.parquet'))
            if sample_files:
                cols = pd.read_parquet(sample_files[0], columns=None).columns
                has_ci = lo_col in cols and hi_col in cols
                break
        if has_ci or sample_files:
            break
    print(f"\nStreaming CO2 data year-by-year (with {ci_level}% CI)...")
    print(f"  Mode: {'MONTHLY (Σ_m CO2_m × ff_m)' if use_monthly else 'ANNUAL (mean CO2 × ff)'}")
    print(f"  {' CI columns found' if has_ci else ' CI columns NOT found — σ will be NaN'}: {lo_col}, {hi_col}")
    print(f"  PRIMARY σ: linear within HUC4, quadrature across HUC4s")
    print(f"  REFERENCE σ: linear within HUC2 (pessimistic, for sensitivity)")

    yearly_huc_flux        = []
    yearly_huc_sigma_huc4  = []
    yearly_huc_sigma_huc2  = []
    yearly_huc_co2         = []
    n_years = 0

    for year in year_range:
        year_dir = co2_dir / f'year={year}'
        if not year_dir.exists():
            print(f"  WARNING: {year_dir} not found, skipping")
            continue

        if use_monthly:
            merged = _merged_monthly(year_dir, lookup, has_ci, lo_col, hi_col)
        else:
            merged = _merged_annual(year_dir, lookup, has_ci, lo_col, hi_col)
        if merged is None:
            continue

        huc_flux, huc_co2, huc_sig4, huc_sig2 = _aggregate_year(merged, has_ci)
        yearly_huc_flux.append(huc_flux)
        yearly_huc_co2.append(huc_co2)
        if has_ci:
            yearly_huc_sigma_huc4.append(huc_sig4)
            yearly_huc_sigma_huc2.append(huc_sig2)

        total = huc_flux.sum()
        n_years += 1
        if n_years <= 3 or n_years % 5 == 0:
            print(f"  Year {year}: total = {total:.4f} Tg C/yr")

        del merged; gc.collect()

    print(f"\n  Averaging across {n_years} years...")
    mean_flux = pd.DataFrame(yearly_huc_flux).T.mean(axis=1)
    mean_co2  = pd.DataFrame(yearly_huc_co2).T.mean(axis=1)

    if has_ci and yearly_huc_sigma_huc4:
        mean_sigma_huc4 = pd.DataFrame(yearly_huc_sigma_huc4).T.mean(axis=1)
        mean_sigma_huc2 = pd.DataFrame(yearly_huc_sigma_huc2).T.mean(axis=1)
    else:
        mean_sigma_huc4 = pd.Series(np.nan, index=mean_flux.index)
        mean_sigma_huc2 = pd.Series(np.nan, index=mean_flux.index)

    return mean_flux, mean_co2, mean_sigma_huc4, mean_sigma_huc2, n_years


# ============================================================
# RUN
# ============================================================
def run_gw_flux(config_name):
    cfg = CONFIGS[config_name]

    print("=" * 90)
    print(f"GROUNDWATER CO2 FLUX BY HUC2 — {config_name} ({tag})  [with uncertainty]")
    print(f"  Source : {co2_predictions}")
    print(f"  Units  : {'ppm  Henrys law' if IS_PPM else 'mol/L (aqueous)'}")
    print(f"  CI     : {CI_LEVEL}%")
    print(f"  Mode   : {'MONTHLY (seasonal covariance)' if USE_MONTHLY else 'ANNUAL (baseline)'}")
    print(f"  Runoff : {cfg['runoff_path'].name}")
    if USE_MONTHLY:
        print(f"  Monthly: {Path(cfg['monthly_runoff_path']).name}")
    print(f"  Years  : {cfg['year_range'].start}–{cfg['year_range'].stop - 1}")
    print(f"  Output : {cfg['output_path'].name}")
    print(f"  σ aggregation: PRIMARY = linear within HUC4, quadrature across HUC4s")
    print(f"                 REFERENCE = linear within HUC2 (sensitivity)")
    print("=" * 90)

    lookup = build_cell_lookup(
        CO2_DIR, cfg['runoff_path'], HUC2_PATH, HUC4_PATH, cfg['year_range'],
        monthly_runoff_path=cfg.get('monthly_runoff_path'), use_monthly=USE_MONTHLY)

    mean_flux, mean_co2, mean_sigma_huc4, mean_sigma_huc2, n_years = \
        compute_flux_streaming(CO2_DIR, cfg['year_range'], lookup,
                               ci_level=CI_LEVEL, use_monthly=USE_MONTHLY)

    #  CONUS σ under both aggregation schemes
    total_flux = mean_flux.sum()

    per_huc2_sigma_huc4 = mean_sigma_huc4.dropna()
    conus_sigma_huc4 = float(np.sqrt((per_huc2_sigma_huc4**2).sum()))

    per_huc2_sigma_huc2 = mean_sigma_huc2.dropna()
    conus_sigma_huc2 = float(np.sqrt((per_huc2_sigma_huc2**2).sum()))

    conus_sigma_linear = float(per_huc2_sigma_huc2.sum())

    results = pd.DataFrame({
        'GW_CO2_flux_TgC_yr':           mean_flux,
        'GW_CO2_flux_sigma_huc4_TgC_yr': mean_sigma_huc4,
        'GW_CO2_flux_sigma_huc2_TgC_yr': mean_sigma_huc2,
    })

    huc_stats = lookup.reset_index().groupby('huc2').agg(
        mean_runoff_m_yr=('runoff_m_yr', 'mean'),
        area_km2=('cell_area_m2', lambda x: x.sum() / 1e6),
        n_cells=('flux_factor', 'count'),
        n_huc4s=('huc4', 'nunique'),
    )
    results = results.join(huc_stats)

    if IS_PPM:
        results['mean_pCO2_ppm']  = mean_co2
        results['mean_CO2_mol_L'] = KH_CO2 * results['mean_pCO2_ppm'] * 1e-6
    else:
        results['mean_CO2_mol_L'] = mean_co2
        results['mean_pCO2_ppm']  = results['mean_CO2_mol_L'] / (KH_CO2 * 1e-6)

    results['mean_CO2_mg_L'] = results['mean_CO2_mol_L'] * MW_CO2 * 1000.0
    results['weighted_yield_gC_m2_yr'] = (
        results['GW_CO2_flux_TgC_yr'] / TG_PER_G / (results['area_km2'] * 1e6)
    )

    #  Butman: load per-HUC, rescale ~95%  90% CI
    BUTMAN_PATH_LOCAL = BASE_DIR / 'runoff/aquatic_carbon_butman.csv'
    if BUTMAN_PATH_LOCAL.exists():
        butman = load_butman(BUTMAN_PATH_LOCAL)
        butman_central = butman['Stream efflux_central']
        butman_lo_95   = butman['Stream efflux_lower']
        butman_hi_95   = butman['Stream efflux_upper']

        results['Butman_stream_central_TgC'] = butman_central

        butman_hw_95 = (butman_hi_95 - butman_lo_95) / 2.0
        butman_hw_90 = butman_hw_95 * (1.645 / 1.960)

        results['Butman_stream_lower_TgC']    = butman_central - butman_hw_90
        results['Butman_stream_upper_TgC']    = butman_central + butman_hw_90
        results['Butman_stream_lower_95_TgC'] = butman_lo_95
        results['Butman_stream_upper_95_TgC'] = butman_hi_95

        results['GW_pct_of_Butman'] = (
            results['GW_CO2_flux_TgC_yr'] / results['Butman_stream_central_TgC'] * 100
        )

    results = results.sort_index()
    results.index.name = 'HUC'

    #  Print summary
    print(f"\n{'='*100}")
    print(f"SUMMARY: GW CO2 Flux by HUC2  [{CI_LEVEL}% CI]  "
          f"({'MONTHLY' if USE_MONTHLY else 'ANNUAL'})")
    print(f"  PRIMARY σ:   linear within HUC4, quadrature across HUC4s")
    print(f"  REFERENCE σ: linear within HUC2, quadrature across HUC2s (pessimistic)")
    print(f"{'='*100}")

    print(f"\n{'HUC':>4}  {'GW flux':>10}  {'σ_HUC4':>10}  {'σ_HUC2':>10}  {'n_HUC4s':>7}  {'CO2 mg/L':>10}  {'Runoff':>8}")
    print(f"{'':>4}  {'TgC/yr':>10}  {'TgC/yr':>10}  {'TgC/yr':>10}  {'':>7}  {'':>10}  {'m/yr':>8}")
    print("-" * 70)
    for huc_id, row in results.iterrows():
        print(
            f"{huc_id:>4}  "
            f"{row['GW_CO2_flux_TgC_yr']:>10.4f}  "
            f"{row.get('GW_CO2_flux_sigma_huc4_TgC_yr', np.nan):>10.4f}  "
            f"{row.get('GW_CO2_flux_sigma_huc2_TgC_yr', np.nan):>10.4f}  "
            f"{row.get('n_huc4s', 0):>7.0f}  "
            f"{row['mean_CO2_mg_L']:>10.1f}  "
            f"{row['mean_runoff_m_yr']:>8.4f}"
        )

    print(f"\n  CONUS totals:")
    print(f"    GW flux central:                    {total_flux:.4f} TgC/yr")
    print(f"    σ_HUC4 (PRIMARY, quad across HUC4s): {conus_sigma_huc4:.4f} TgC/yr  ({conus_sigma_huc4/total_flux*100:.1f}% rel)")
    print(f"    σ_HUC2 (quad across HUC2s):           {conus_sigma_huc2:.4f} TgC/yr  ({conus_sigma_huc2/total_flux*100:.1f}% rel)")
    print(f"    σ_linear (all cells correlated):      {conus_sigma_linear:.4f} TgC/yr  ({conus_sigma_linear/total_flux*100:.1f}% rel)  [pessimistic ref]")
    print(f"\n  90% CI (PRIMARY, HUC4 block):")
    print(f"    [{total_flux - Z_90*conus_sigma_huc4:.4f}, {total_flux + Z_90*conus_sigma_huc4:.4f}] TgC/yr")
    print(f"\n  NOTE: lower/upper NOT written to CSV.")
    print(f"        Downstream computes CI from GW_CO2_flux_sigma_huc4_TgC_yr (primary).")

    #  Save
    results.to_csv(cfg['output_path'])
    print(f"\nSaved: {cfg['output_path']}")
    print(f"  Columns written:")
    print(f"    GW_CO2_flux_TgC_yr               — central estimate")
    print(f"    GW_CO2_flux_sigma_huc4_TgC_yr    — PRIMARY: per-HUC2 σ from HUC4 block")
    print(f"    GW_CO2_flux_sigma_huc2_TgC_yr    — REFERENCE: per-HUC2 linear-within-HUC2 σ")

    cell_cols = ['lat', 'lon', 'huc2', 'huc4', 'runoff_m_yr', 'cell_area_m2', 'flux_factor']
    if USE_MONTHLY and MONTH_COLS[0] in lookup.columns:
        cell_cols += MONTH_COLS
    cell_out = lookup.reset_index()[cell_cols]
    cell_out.to_csv(cfg['cell_output_path'], index=False)
    print(f"Cell lookup saved: {cfg['cell_output_path']}")

    return results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == '__main__':
    results_sl = run_gw_flux('saccardi_liu')
    results_b  = run_gw_flux('butman')
