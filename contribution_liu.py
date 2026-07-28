"""
contribution_liu.py

Liu et al. (2022) stream CO2 efflux with ephemeral and ice corrections on the
NHD-HR network, and the groundwater contribution percentage against Liu, Saccardi,
and Butman efflux (the values reported in the manuscript and Figure S4).
Requires gw_co2_flux_by_huc2_*.csv from gw_flux.py.
"""
co2_predictions = "co2_predictions_aq_uncertainty2"

import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import json
from pathlib import Path
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
tag = 'aq' if co2_predictions.endswith('_aq_uncertainty2') else 'ppm'

BASE_DIR       = Path("data")
LIU_DIR        = BASE_DIR / 'runoff' / 'liu_2022'
HUC_PATH       = BASE_DIR / 'shp_files/Watershed_Boundary_Dataset_HUC_2.gpkg'
BUTMAN_PATH    = BASE_DIR / 'runoff/aquatic_carbon_butman.csv'
SACCARDI_PATH  = BASE_DIR / f'saccardi_stream_co2_by_huc2_{tag}.csv'
GW_FLUX_PATH   = BASE_DIR / f'gw_co2_flux_by_huc2_1979_2014_{tag}.csv'
CLIMATE_JSON_PATH = BASE_DIR / 'runoff/huc2_monthly_climate_1971_2000.json'
OUTPUT_PATH    = BASE_DIR / f'liu_stream_co2_by_huc2_{tag}.csv'

CONUS_HUC2 = [str(i).zfill(2) for i in range(1, 19)]
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
TG_PER_G = 1e-12
Z_90 = 1.645  # used for GW bounds only

#  Liu et al. Table S2: monthly 1σ errors (Tg C, GLOBAL)
LIU_MONTHLY_1SIGMA_GLOBAL    = [20, 20, 30, 30, 20, 20, 10, 20, 20, 10, 20, 20]
LIU_MONTHLY_EMISSION_GLOBAL  = [115, 112, 137, 174, 208, 209, 205, 203, 174, 144, 114, 112]
LIU_ANNUAL_EMISSION_GLOBAL   = 1970
LIU_ANNUAL_1SIGMA_GLOBAL     = 200
LIU_MONTHLY_REL_SIGMA = [
    s / e for s, e in zip(LIU_MONTHLY_1SIGMA_GLOBAL, LIU_MONTHLY_EMISSION_GLOBAL)
]
LIU_ANNUAL_REL_SIGMA = LIU_ANNUAL_1SIGMA_GLOBAL / LIU_ANNUAL_EMISSION_GLOBAL

# Saccardi published ±23 TgC/yr
SACCARDI_PUBLISHED_DF = 23.0


# ============================================================
# HELPER: Load HUC2 watershed areas
# ============================================================
def load_huc2_areas(huc_path, conus_huc2):
    hucs = gpd.read_file(huc_path)
    hucs = hucs[hucs['huc2'].isin(conus_huc2)].copy()
    return hucs.set_index('huc2')['areasqkm']


# ============================================================
# EPHEMERAL CORRECTION (Ray 2013)
# ============================================================
def compute_effective_area_ray2013_monthly(saccardi_path, climate_json_path):
    saccardi = pd.read_csv(saccardi_path, index_col='HUC', dtype={'HUC': str})

    with open(climate_json_path, 'r') as f:
        huc_climate = json.load(f)

    ray_coeffs = {
        1: {'a': -0.009, 'b': 0.029, 'c': 0.77,  'd': -0.0019, 'e': 0.017, 'f': 0.026},
        2: {'a': -0.009, 'b': 0.029, 'c': 0.61,  'd': -0.0021, 'e': 0.011, 'f': 0.088},
        3: {'a': -0.008, 'b': 0.028, 'c': 0.44,  'd': -0.0018, 'e': 0.011, 'f': 0.058},
        4: {'a': -0.005, 'b': 0.023, 'c': 0.27,  'd': -0.0028, 'e': 0.012, 'f': 0.127},
    }

    results_list = []
    for huc2 in saccardi.index:
        if huc2 not in huc_climate:
            continue
        clim = huc_climate[huc2]
        total_area = saccardi.loc[huc2, 'stream_area_km2']

        so_areas_km2 = {}
        for so in range(1, 5):
            col = f'area_SO{so}_m2'
            if col in saccardi.columns and pd.notna(saccardi.loc[huc2, col]):
                so_areas_km2[so] = saccardi.loc[huc2, col] / 1e6
            else:
                frac = {1: 0.30, 2: 0.20, 3: 0.15, 4: 0.10}[so]
                so_areas_km2[so] = total_area * frac

        monthly_effective = []
        monthly_ephem = []
        for m_idx in range(12):
            temp_m = clim['temp'][m_idx]
            prec_m = clim['prec'][m_idx]
            month_ephem_area = 0.0
            for so in range(1, 5):
                rc = ray_coeffs[so]
                percInterm = np.clip(rc['a'] * prec_m + rc['b'] * temp_m + rc['c'], 0.0, 0.9)
                timedryout = np.clip(rc['d'] * prec_m + rc['e'] * temp_m + rc['f'], 0.0, 0.9)
                month_ephem_area += so_areas_km2[so] * percInterm * timedryout
            monthly_effective.append(total_area - month_ephem_area)
            monthly_ephem.append(month_ephem_area)

        annual_effective = np.mean(monthly_effective)
        annual_ephem = np.mean(monthly_ephem)
        pct_reduction = (annual_ephem / total_area * 100) if total_area > 0 else 0

        results_list.append({
            'huc2': huc2,
            'total_area_km2': total_area,
            'annual_ephem_area_km2': annual_ephem,
            'annual_effective_area_km2': annual_effective,
            'pct_reduction': pct_reduction,
        })

    eph_df = pd.DataFrame(results_list).set_index('huc2')

    print("\n  Ray 2013 Monthly Ephemeral Correction:")
    print(f"  {'HUC':>4}  {'Total km²':>10}  {'Ephem km²':>10}  {'Effect km²':>11}  {'Reduction':>10}")
    print(f"  {'-'*55}")
    for huc2, row in eph_df.iterrows():
        print(f"  {huc2:>4}  {row['total_area_km2']:>10.0f}  {row['annual_ephem_area_km2']:>10.0f}  "
              f"{row['annual_effective_area_km2']:>11.0f}  {row['pct_reduction']:>9.1f}%")
    tot = eph_df['total_area_km2'].sum()
    eph = eph_df['annual_ephem_area_km2'].sum()
    print(f"  {'TOTAL':>4}  {tot:>10.0f}  {eph:>10.0f}  {tot-eph:>11.0f}  {eph/tot*100:>9.1f}%")
    return eph_df


# ============================================================
# ICE CORRECTION
# ============================================================
ERA5_PARQUET_DIR = Path("data/era5_month_parquet_01deg")
ICE_THRESHOLD_C = -4.0
ICE_MELT_EMISSION_FRAC = 0.17

def compute_ice_correction_by_huc2(era5_dir, huc_path, year_range=range(1971, 2001)):
    CONUS_HUC2 = [str(i).zfill(2) for i in range(1, 19)]
    sample = None
    for year in year_range:
        for month in range(1, 13):
            fp = era5_dir / f'year={year}' / f'month={month}' / 'data.parquet'
            if fp.exists():
                sample = pd.read_parquet(fp, columns=['latitude', 'longitude'])
                break
        if sample is not None:
            break

    import geopandas as gpd
    hucs = gpd.read_file(huc_path)
    hucs = hucs[hucs['huc2'].isin(CONUS_HUC2)].to_crs('EPSG:4326')
    geom = gpd.points_from_xy(sample['longitude'], sample['latitude'])
    gdf = gpd.GeoDataFrame(sample, geometry=geom, crs='EPSG:4326')
    joined = gpd.sjoin(gdf, hucs[['huc2', 'geometry']], how='left', predicate='within')
    joined = joined.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
    cell_huc = joined[['latitude', 'longitude', 'huc2']].dropna(subset=['huc2'])
    cell_huc = cell_huc.set_index(['latitude', 'longitude'])
    print(f"  {len(cell_huc)} ERA5 cells assigned to HUC2")

    from collections import defaultdict
    ice_data = defaultdict(lambda: defaultdict(list))
    for year in year_range:
        for month in range(1, 13):
            fp = era5_dir / f'year={year}' / f'month={month}' / 'data.parquet'
            if not fp.exists():
                continue
            df = pd.read_parquet(fp, columns=['latitude', 'longitude', 't2m'])
            df = df.set_index(['latitude', 'longitude'])
            merged = cell_huc[['huc2']].join(df['t2m'], how='inner')
            merged['frozen'] = (merged['t2m'] - 273.15) < ICE_THRESHOLD_C
            for huc2, grp in merged.groupby('huc2'):
                ice_data[huc2][month].append({
                    'n_frozen': grp['frozen'].sum(), 'n_total': len(grp),
                })
        if year % 5 == 0:
            print(f"  Processed through {year}")

    rows = []
    for huc2 in sorted(ice_data.keys()):
        for month in range(1, 13):
            yearly_fracs = [d['n_frozen'] / d['n_total']
                            for d in ice_data[huc2][month] if d['n_total'] > 0]
            rows.append({'huc2': huc2, 'month': month,
                         'ice_fraction': np.mean(yearly_fracs) if yearly_fracs else 0.0})

    ice_df = pd.DataFrame(rows)
    ice_wide = ice_df.pivot(index='huc2', columns='month', values='ice_fraction')
    ice_wide.columns = [f'ice_frac_{m:02d}' for m in ice_wide.columns]
    ice_wide['ice_frac_annual'] = ice_wide.mean(axis=1)

    print(f"\n  {'HUC':>4}  {'Ann':>5}  {'Jan':>5}  {'Apr':>5}  {'Jul':>5}  {'Oct':>5}")
    print(f"  {'-'*35}")
    for huc2, row in ice_wide.iterrows():
        print(f"  {huc2:>4}  {row['ice_frac_annual']:>5.1%}  "
              f"{row['ice_frac_01']:>5.1%}  {row['ice_frac_04']:>5.1%}  "
              f"{row['ice_frac_07']:>5.1%}  {row['ice_frac_10']:>5.1%}")
    return ice_df, ice_wide


# ============================================================
# STEP 1: Build pixel  HUC2 lookup
# ============================================================
def build_pixel_huc_lookup(liu_dir, huc_path):
    print("Building pixel  HUC2 lookup from Jan_co2f.tif...")
    da = rioxarray.open_rasterio(liu_dir / 'Jan_co2f.tif', masked=True).squeeze('band', drop=True)
    if 'x' in da.dims:
        da = da.rename({'x': 'longitude', 'y': 'latitude'})
    da = da.sel(latitude=slice(50.1, 23.9), longitude=slice(-125.1, -65.9))
    df = da.to_dataframe(name='co2f').reset_index().dropna(subset=['co2f'])
    df = df[['latitude', 'longitude']].copy()
    df['latitude'] = df['latitude'].round(6)
    df['longitude'] = df['longitude'].round(6)
    print(f"  {len(df):,} stream pixels in CONUS")

    hucs = gpd.read_file(huc_path)
    hucs = hucs[hucs['huc2'].isin(CONUS_HUC2)].to_crs('EPSG:4326')
    geom = gpd.points_from_xy(df['longitude'], df['latitude'])
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs='EPSG:4326')

    CHUNK = 500_000
    huc_assignments = []
    for i in range(0, len(gdf), CHUNK):
        chunk = gdf.iloc[i:i + CHUNK]
        joined = gpd.sjoin(chunk, hucs[['huc2', 'geometry']], how='left', predicate='within')
        joined = joined.drop_duplicates(subset=['latitude', 'longitude'], keep='first')
        huc_assignments.append(joined[['latitude', 'longitude', 'huc2']])
        print(f"    Chunk {i // CHUNK + 1}/{(len(gdf) // CHUNK) + 1}: {len(chunk):,} pixels")

    lookup = pd.concat(huc_assignments, ignore_index=True)
    lookup = lookup.dropna(subset=['huc2'])
    lookup = lookup.set_index(['latitude', 'longitude'])
    print(f"  {len(lookup):,}/{len(df):,} pixels assigned to HUC2 "
          f"({100 * len(lookup) / len(df):.1f}%)")
    del da, df, gdf; gc.collect()
    return lookup


# ============================================================
# STEP 2: Process monthly GeoTIFFs
# ============================================================
def aggregate_by_huc(liu_dir, lookup):
    print("\nProcessing monthly GeoTIFFs...")

    monthly_co2f = []
    monthly_pco2 = []
    monthly_huc_means = {}

    for month_idx, (mon, n_days) in enumerate(zip(MONTHS, DAYS_IN_MONTH), 1):
        co2f_path = liu_dir / f'{mon}_co2f.tif'
        if not co2f_path.exists():
            continue

        da = rioxarray.open_rasterio(co2f_path, masked=True).squeeze('band', drop=True)
        if 'x' in da.dims:
            da = da.rename({'x': 'longitude', 'y': 'latitude'})
        da = da.sel(latitude=slice(50.1, 23.9), longitude=slice(-125.1, -65.9))
        df_co2f = da.to_dataframe(name='co2f').reset_index().dropna(subset=['co2f'])
        df_co2f['latitude'] = df_co2f['latitude'].round(6)
        df_co2f['longitude'] = df_co2f['longitude'].round(6)
        df_co2f = df_co2f.set_index(['latitude', 'longitude'])

        merged = lookup[['huc2']].join(df_co2f['co2f'], how='inner')
        huc_stats = merged.groupby('huc2').agg(
            mean_co2f=('co2f', 'mean'),
            n_pixels=('co2f', 'count'),
        )
        huc_stats['month'] = mon
        monthly_co2f.append(huc_stats)

        for huc2, row in huc_stats.iterrows():
            if huc2 not in monthly_huc_means:
                monthly_huc_means[huc2] = {}
            monthly_huc_means[huc2][mon] = row['mean_co2f']

        pco2_path = liu_dir / f'co2_{month_idx:02d}.tif'
        if pco2_path.exists():
            da_p = rioxarray.open_rasterio(pco2_path, masked=True).squeeze('band', drop=True)
            if 'x' in da_p.dims:
                da_p = da_p.rename({'x': 'longitude', 'y': 'latitude'})
            da_p = da_p.sel(latitude=slice(50.1, 23.9), longitude=slice(-125.1, -65.9))
            df_pco2 = da_p.to_dataframe(name='pco2').reset_index().dropna(subset=['pco2'])
            df_pco2['latitude'] = df_pco2['latitude'].round(6)
            df_pco2['longitude'] = df_pco2['longitude'].round(6)
            df_pco2 = df_pco2.set_index(['latitude', 'longitude'])
            merged_p = lookup[['huc2']].join(df_pco2['pco2'], how='inner')
            monthly_pco2.append(merged_p.groupby('huc2')['pco2'].mean().rename(f'pco2_{mon}'))
            del da_p, df_pco2, merged_p

        print(f"  {mon}: {len(merged):,} pixels, mean co2f = {merged['co2f'].mean():.1f} g C m⁻² yr⁻¹")
        del da, df_co2f, merged; gc.collect()

    all_co2f = pd.concat(monthly_co2f)
    results = all_co2f.groupby('huc2').agg(
        mean_co2f_gC_m2_yr=('mean_co2f', 'mean'),
        mean_stream_pixels=('n_pixels', 'mean'),
    )
    if monthly_pco2:
        results['mean_stream_pco2_uatm'] = pd.concat(monthly_pco2, axis=1).mean(axis=1)

    monthly_df = pd.DataFrame(monthly_huc_means).T
    monthly_df.index.name = 'huc2'

    return results, monthly_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 110)
    print("STREAM CO2 EFFLUX BY HUC2 — Liu et al. 2022")
    print("  Flux: co2f rate × NHD stream area (corrected for ephemeral + ice)")
    print("  Liu uncertainty: monthly 1σ from Table S2, propagated as reported (no CI scaling)")
    print("  Saccardi uncertainty: published ±23 TgC/yr (flux-proportional)")
    print("  Butman uncertainty: ~95% CI rescaled to 90% CI (×1.645/1.960)")
    print("  GW uncertainty: HUC4-block aggregation (linear within HUC4, quadrature across HUC4s)")
    print("  With Ray 2013 ephemeral + Liu ice corrections")
    print("=" * 110)

    print("\n  Liu et al. Table S2 monthly relative 1σ errors:")
    print(f"  {'Month':<5} {'Global Tg C':>10} {'1σ Tg C':>8} {'Rel σ':>8}")
    for i, mon in enumerate(MONTHS):
        print(f"  {mon:<5} {LIU_MONTHLY_EMISSION_GLOBAL[i]:>10} "
              f"{LIU_MONTHLY_1SIGMA_GLOBAL[i]:>8} "
              f"{LIU_MONTHLY_REL_SIGMA[i]:>8.1%}")
    print(f"  {'Annual':<5} {LIU_ANNUAL_EMISSION_GLOBAL:>10} "
          f"{LIU_ANNUAL_1SIGMA_GLOBAL:>8} "
          f"{LIU_ANNUAL_REL_SIGMA:>8.1%}")

    lookup = build_pixel_huc_lookup(LIU_DIR, HUC_PATH)
    results, monthly_df = aggregate_by_huc(LIU_DIR, lookup)

    huc2_areas = load_huc2_areas(HUC_PATH, CONUS_HUC2)
    results['huc2_area_km2'] = huc2_areas
    results['huc2_area_m2']  = huc2_areas * 1e6

    if SACCARDI_PATH.exists():
        saccardi = pd.read_csv(SACCARDI_PATH, index_col='HUC', dtype={'HUC': str})
        results['stream_area_km2'] = saccardi['stream_area_km2']
        results['Saccardi_flux_TgC_yr'] = saccardi['Saccardi_flux_TgC_yr']

        if 'Saccardi_flux_lower_TgC_yr' in saccardi.columns:
            results['Saccardi_flux_lower_TgC_yr'] = saccardi['Saccardi_flux_lower_TgC_yr']
            results['Saccardi_flux_upper_TgC_yr'] = saccardi['Saccardi_flux_upper_TgC_yr']
            print("   Saccardi per-HUC flux bounds loaded (±δF from published ±23)")
        else:
            results['Saccardi_flux_lower_TgC_yr'] = np.nan
            results['Saccardi_flux_upper_TgC_yr'] = np.nan
            print("   Saccardi flux bounds not in CSV — will fall back to GW-only uncertainty")

        results['Liu_flux_TgC_yr'] = (
            results['mean_co2f_gC_m2_yr'] *
            results['stream_area_km2'] * 1e6 * TG_PER_G
        )

        if CLIMATE_JSON_PATH.exists():
            eph = compute_effective_area_ray2013_monthly(SACCARDI_PATH, CLIMATE_JSON_PATH)
            results['effective_area_km2'] = eph['annual_effective_area_km2']
            results['ephem_pct_reduction'] = eph['pct_reduction']

            results['Liu_flux_corrected_TgC_yr'] = (
                results['mean_co2f_gC_m2_yr'] *
                results['effective_area_km2'] * 1e6 * TG_PER_G
            )

            print(f"\n  HUC2 watershed area total:       {results['huc2_area_km2'].sum():,.0f} km²")
            print(f"  Total stream area (raw):          {results['stream_area_km2'].sum():,.0f} km²")
            print(f"  Total effective area (Ray2013):   {results['effective_area_km2'].sum():,.0f} km²")
            print(f"  Liu flux (raw area):              {results['Liu_flux_TgC_yr'].sum():.2f} TgC/yr")
            print(f"  Liu flux (ephemeral corrected):   {results['Liu_flux_corrected_TgC_yr'].sum():.2f} TgC/yr")

            print("\nComputing ice correction from ERA5 t2m...")
            ice_df, ice_wide = compute_ice_correction_by_huc2(
                ERA5_PARQUET_DIR, HUC_PATH, year_range=range(1971, 2001)
            )
            MONTHS_NUM = list(range(1, 13))

            for huc2 in results.index:
                if huc2 not in ice_wide.index:
                    results.loc[huc2, 'active_area_km2'] = results.loc[huc2, 'effective_area_km2']
                    results.loc[huc2, 'ice_frac_annual'] = 0.0
                    results.loc[huc2, 'n_melt_months'] = 0
                    continue

                ephem_area = results.loc[huc2, 'effective_area_km2']
                monthly_active = []
                for m in MONTHS_NUM:
                    ice_frac = ice_wide.loc[huc2, f'ice_frac_{m:02d}']
                    monthly_active.append(ephem_area * (1.0 - ice_frac))

                results.loc[huc2, 'active_area_km2'] = np.mean(monthly_active)
                results.loc[huc2, 'ice_frac_annual'] = ice_wide.loc[huc2, 'ice_frac_annual']

                ice_fracs = [ice_wide.loc[huc2, f'ice_frac_{m:02d}'] for m in MONTHS_NUM]
                melt_months = sum(1 for i in range(1, 12) if ice_fracs[i] < ice_fracs[i - 1])
                results.loc[huc2, 'n_melt_months'] = melt_months

            results['Liu_flux_ice_corrected_TgC_yr'] = (
                results['mean_co2f_gC_m2_yr'] *
                results['active_area_km2'] * 1e6 * TG_PER_G
            )
            results['ice_melt_emission_TgC_yr'] = (
                results['mean_co2f_gC_m2_yr'] *
                results['active_area_km2'] * 1e6 * TG_PER_G *
                ICE_MELT_EMISSION_FRAC * results['n_melt_months'] / 12.0
            )
            results['Liu_flux_final_TgC_yr'] = (
                results['Liu_flux_ice_corrected_TgC_yr'] +
                results['ice_melt_emission_TgC_yr']
            )

            print(f"  Total active area (ephem+ice):    {results['active_area_km2'].sum():,.0f} km²")
            print(f"  Liu flux (ephem+ice corrected):   {results['Liu_flux_ice_corrected_TgC_yr'].sum():.2f} TgC/yr")
            print(f"  Ice-melt emissions added back:    {results['ice_melt_emission_TgC_yr'].sum():.4f} TgC/yr")
            print(f"  Liu flux (final):                 {results['Liu_flux_final_TgC_yr'].sum():.2f} TgC/yr")

            #  Liu uncertainty: propagate monthly 1σ as reported
            print("\n  Propagating Liu monthly 1σ uncertainty (as reported, no CI scaling)...")
            print("  Method: all-linear (months correlated within HUC, HUCs correlated across CONUS)")

            for huc2 in results.index:
                area_m2 = results.loc[huc2, 'active_area_km2'] * 1e6
                monthly_sigma_linear = 0.0
                for m_idx, mon in enumerate(MONTHS):
                    co2f_m = (monthly_df.loc[huc2, mon]
                              if huc2 in monthly_df.index and mon in monthly_df.columns
                              else results.loc[huc2, 'mean_co2f_gC_m2_yr'])
                    flux_m = co2f_m * area_m2 * TG_PER_G / 12.0
                    sigma_m = flux_m * LIU_MONTHLY_REL_SIGMA[m_idx]
                    monthly_sigma_linear += sigma_m
                results.loc[huc2, 'Liu_flux_sigma_TgC_yr'] = monthly_sigma_linear

            total_fin_tmp = results['Liu_flux_final_TgC_yr'].sum()
            conus_sigma = results['Liu_flux_sigma_TgC_yr'].sum()
            conus_sigma_annual = total_fin_tmp * LIU_ANNUAL_REL_SIGMA

            print(f"\n  Liu uncertainty estimates (1σ, as reported):")
            print(f"  {'Method':<65} {'CONUS 1σ':>9} {'Rel':>8}")
            print(f"  {'-'*85}")
            for label, sigma in [
                ('All-linear (monthly linear within HUC, HUC2s summed linearly)', conus_sigma),
                ('Annual relative (200/1970 = 10.2%) × CONUS total',              conus_sigma_annual),
            ]:
                print(f"  {label:<65} {sigma:>8.2f}  {sigma/total_fin_tmp:>7.1%}")
            print(f"  {'Liu published (global): 1970 ± 200 TgC/yr (1σ)':<65} {'200':>8}  {'10.2%':>7}")

            print(f"\n   Liu CONUS = {total_fin_tmp:.2f} ± {conus_sigma:.2f} TgC/yr (1σ)")

            results['Liu_flux_lower_TgC_yr'] = (
                results['Liu_flux_final_TgC_yr'] - results['Liu_flux_sigma_TgC_yr']
            )
            results['Liu_flux_upper_TgC_yr'] = (
                results['Liu_flux_final_TgC_yr'] + results['Liu_flux_sigma_TgC_yr']
            )

        else:
            print(f"\n   Climate JSON not found, skipping corrections")
            results['effective_area_km2'] = results['stream_area_km2']
            results['active_area_km2'] = results['stream_area_km2']
            results['Liu_flux_corrected_TgC_yr'] = results['Liu_flux_TgC_yr']
            results['Liu_flux_final_TgC_yr'] = results['Liu_flux_TgC_yr']
            results['ice_frac_annual'] = 0.0
            results['Liu_flux_sigma_TgC_yr'] = results['Liu_flux_TgC_yr'] * LIU_ANNUAL_REL_SIGMA
            results['Liu_flux_lower_TgC_yr'] = (
                results['Liu_flux_TgC_yr'] - results['Liu_flux_sigma_TgC_yr']
            )
            results['Liu_flux_upper_TgC_yr'] = (
                results['Liu_flux_TgC_yr'] + results['Liu_flux_sigma_TgC_yr']
            )
    else:
        print(f"\n   Saccardi not found")
        for col in ['Liu_flux_TgC_yr', 'Liu_flux_corrected_TgC_yr',
                    'Liu_flux_final_TgC_yr', 'Liu_flux_sigma_TgC_yr',
                    'Liu_flux_lower_TgC_yr', 'Liu_flux_upper_TgC_yr']:
            results[col] = np.nan

    #  Join GW flux
    conus_sigma_huc4_gw = np.nan
    conus_sigma_huc2_gw = np.nan
    if GW_FLUX_PATH.exists():
        gw = pd.read_csv(GW_FLUX_PATH, index_col='HUC', dtype={'HUC': str})
        results['GW_CO2_flux_TgC_yr'] = gw['GW_CO2_flux_TgC_yr']
        total_gw_central = results['GW_CO2_flux_TgC_yr'].sum()

        if 'GW_CO2_flux_sigma_huc4_TgC_yr' in gw.columns:
            results['GW_CO2_flux_sigma_huc4_TgC_yr'] = gw['GW_CO2_flux_sigma_huc4_TgC_yr']
            per_huc_sigma_huc4 = results['GW_CO2_flux_sigma_huc4_TgC_yr'].dropna()
            conus_sigma_huc4_gw = float(np.sqrt((per_huc_sigma_huc4**2).sum()))
            print(f"  GW HUC4-block σ: {conus_sigma_huc4_gw:.4f} TgC/yr "
                  f"({conus_sigma_huc4_gw/total_gw_central*100:.1f}% rel)  ◄ PRIMARY")
        else:
            results['GW_CO2_flux_sigma_huc4_TgC_yr'] = np.nan
            print("   GW HUC4 σ not in CSV — falling back to HUC2 σ if available")

        if 'GW_CO2_flux_sigma_huc2_TgC_yr' in gw.columns:
            results['GW_CO2_flux_sigma_huc2_TgC_yr'] = gw['GW_CO2_flux_sigma_huc2_TgC_yr']
            per_huc_sigma_huc2 = results['GW_CO2_flux_sigma_huc2_TgC_yr'].dropna()
            conus_sigma_huc2_gw = float(np.sqrt((per_huc_sigma_huc2**2).sum()))
            print(f"  GW HUC2-block σ: {conus_sigma_huc2_gw:.4f} TgC/yr "
                  f"({conus_sigma_huc2_gw/total_gw_central*100:.1f}% rel)  [sensitivity]")
        else:
            results['GW_CO2_flux_sigma_huc2_TgC_yr'] = np.nan

        if not np.isnan(conus_sigma_huc4_gw):
            _conus_sigma_for_per_huc = conus_sigma_huc4_gw
        elif not np.isnan(conus_sigma_huc2_gw):
            _conus_sigma_for_per_huc = conus_sigma_huc2_gw
        else:
            _conus_sigma_for_per_huc = np.nan

        if not np.isnan(_conus_sigma_for_per_huc) and total_gw_central > 0:
            _scale_lo = max(0.0, total_gw_central - Z_90 * _conus_sigma_for_per_huc) / total_gw_central
            _scale_hi = (total_gw_central + Z_90 * _conus_sigma_for_per_huc) / total_gw_central
            results['GW_CO2_flux_lower_TgC_yr'] = results['GW_CO2_flux_TgC_yr'] * _scale_lo
            results['GW_CO2_flux_upper_TgC_yr'] = results['GW_CO2_flux_TgC_yr'] * _scale_hi
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

        print(f"  Butman bounds rescaled ~95%  90% CI:")
        print(f"    95%: [{butman_lo_95.sum():.1f}, {butman_hi_95.sum():.1f}]")
        print(f"    90%: [{(butman_central - butman_hw_90).sum():.1f}, "
              f"{(butman_central + butman_hw_90).sum():.1f}]")

    #  Watershed-normalized yields
    results['GW_yield_gC_m2_yr']       = results['GW_CO2_flux_TgC_yr']        / TG_PER_G / results['huc2_area_m2']
    results['Liu_yield_gC_m2_yr']      = results['Liu_flux_final_TgC_yr']     / TG_PER_G / results['huc2_area_m2']
    results['Saccardi_yield_gC_m2_yr'] = results['Saccardi_flux_TgC_yr']      / TG_PER_G / results['huc2_area_m2']
    results['Butman_yield_gC_m2_yr']   = results['Butman_stream_central_TgC'] / TG_PER_G / results['huc2_area_m2']

    results['GW_pct_of_Liu_final'] = results['GW_yield_gC_m2_yr'] / results['Liu_yield_gC_m2_yr']      * 100
    results['GW_pct_of_Saccardi']  = results['GW_yield_gC_m2_yr'] / results['Saccardi_yield_gC_m2_yr'] * 100
    results['GW_pct_of_Butman']    = results['GW_yield_gC_m2_yr'] / results['Butman_yield_gC_m2_yr']   * 100

    #  Per-HUC % bounds
    if 'GW_CO2_flux_lower_TgC_yr' in results.columns:
        if not np.isnan(conus_sigma_huc4_gw):
            conus_sigma_for_pct = conus_sigma_huc4_gw
        elif not np.isnan(conus_sigma_huc2_gw):
            conus_sigma_for_pct = conus_sigma_huc2_gw
        else:
            conus_sigma_for_pct = np.nan

        total_gw_central = results['GW_CO2_flux_TgC_yr'].sum()
        if not np.isnan(conus_sigma_for_pct) and total_gw_central > 0:
            scale_lo = max(0.0, total_gw_central - Z_90 * conus_sigma_for_pct) / total_gw_central
            scale_hi = (total_gw_central + Z_90 * conus_sigma_for_pct) / total_gw_central
        else:
            scale_lo = np.nan
            scale_hi = np.nan

        gw_yield_lo = results['GW_yield_gC_m2_yr'] * scale_lo
        gw_yield_hi = results['GW_yield_gC_m2_yr'] * scale_hi
        results['GW_yield_lower_gC_m2_yr'] = gw_yield_lo
        results['GW_yield_upper_gC_m2_yr'] = gw_yield_hi

        # Liu per-HUC yield bounds (1σ)
        liu_yield_lower = results['Liu_flux_lower_TgC_yr'] / TG_PER_G / results['huc2_area_m2']
        liu_yield_upper = results['Liu_flux_upper_TgC_yr'] / TG_PER_G / results['huc2_area_m2']
        but_yield_lower = results['Butman_stream_lower_TgC'] / TG_PER_G / results['huc2_area_m2']
        but_yield_upper = results['Butman_stream_upper_TgC'] / TG_PER_G / results['huc2_area_m2']

        results['GW_pct_of_Liu_lower']    = gw_yield_lo / liu_yield_upper * 100
        results['GW_pct_of_Liu_upper']    = gw_yield_hi / liu_yield_lower * 100
        results['GW_pct_of_Butman_lower'] = gw_yield_lo / but_yield_upper * 100
        results['GW_pct_of_Butman_upper'] = gw_yield_hi / but_yield_lower * 100

        # Saccardi per-HUC yield bounds (published ±23)
        if 'Saccardi_flux_lower_TgC_yr' in results.columns and \
           results['Saccardi_flux_lower_TgC_yr'].notna().any():
            sac_yield_lower = results['Saccardi_flux_lower_TgC_yr'] / TG_PER_G / results['huc2_area_m2']
            sac_yield_upper = results['Saccardi_flux_upper_TgC_yr'] / TG_PER_G / results['huc2_area_m2']
            results['GW_pct_of_Saccardi_lower'] = gw_yield_lo / sac_yield_upper * 100
            results['GW_pct_of_Saccardi_upper'] = gw_yield_hi / sac_yield_lower * 100
            print("   Saccardi % bounds: asymmetric (Saccardi ±δF + GW block σ)")
        elif 'Saccardi_yield_gC_m2_yr' in results.columns:
            sac_yield = results['Saccardi_yield_gC_m2_yr']
            results['GW_pct_of_Saccardi_lower'] = gw_yield_lo / sac_yield * 100
            results['GW_pct_of_Saccardi_upper'] = gw_yield_hi / sac_yield * 100
            print("   Saccardi % bounds: GW uncertainty only (Saccardi bounds unavailable)")

    results.index.name = 'HUC'
    results = results.sort_index()

    #  Print results table
    print("\n" + "=" * 180)
    print("RESULTS: Stream CO2 Efflux by HUC2 — Liu (2022) × Saccardi area")
    print("  Liu bounds: 1σ (as reported in Table S2)")
    print("  GW bounds: 90% CI (HUC4-block σ)")
    print("=" * 180)
    print(f"{'HUC':>4}  "
          f"{' Liu final (TgC/yr) ':>34}  "
          f"{' GW flux (TgC/yr) ':>34}  "
          f"{' GW / Liu (%) ':>28}  "
          f"{'Ice%':>6}  {'Rate':>10}  {'Act km²':>8}")
    print(f"{'':>4}  "
          f"{'lower':>10} {'central':>10} {'upper':>10}  "
          f"{'lower':>10} {'central':>10} {'upper':>10}  "
          f"{'lower':>8} {'central':>8} {'upper':>8}  "
          f"{'':>6}  {'gC/m²/yr':>10}  {'':>8}")
    print(f"{'':>4}  "
          f"{'(1σ)':>10} {'':>10} {'(1σ)':>10}  "
          f"{'(90%CI)':>10} {'':>10} {'(90%CI)':>10}  "
          f"{'':>8} {'':>8} {'':>8}  "
          f"{'':>6}  {'':>10}  {'':>8}")
    print("-" * 180)

    for huc_id, row in results.iterrows():
        print(
            f"{huc_id:>4}  "
            f"{row.get('Liu_flux_lower_TgC_yr', np.nan):>10.3f} "
            f"{row.get('Liu_flux_final_TgC_yr', np.nan):>10.3f} "
            f"{row.get('Liu_flux_upper_TgC_yr', np.nan):>10.3f}  "
            f"{row.get('GW_CO2_flux_lower_TgC_yr', np.nan):>10.4f} "
            f"{row.get('GW_CO2_flux_TgC_yr', np.nan):>10.4f} "
            f"{row.get('GW_CO2_flux_upper_TgC_yr', np.nan):>10.4f}  "
            f"{row.get('GW_pct_of_Liu_lower', np.nan):>7.1f}% "
            f"{row.get('GW_pct_of_Liu_final', np.nan):>7.1f}% "
            f"{row.get('GW_pct_of_Liu_upper', np.nan):>7.1f}%  "
            f"{row.get('ice_frac_annual', 0)*100:>5.1f}%  "
            f"{row['mean_co2f_gC_m2_yr']:>10.1f}  "
            f"{row.get('active_area_km2', np.nan):>8.0f}"
        )

    #  Totals
    print("-" * 180)
    total_raw  = results.get('Liu_flux_TgC_yr',           pd.Series(dtype=float)).sum()
    total_eph  = results.get('Liu_flux_corrected_TgC_yr', pd.Series(dtype=float)).sum()
    total_fin  = results.get('Liu_flux_final_TgC_yr',     pd.Series(dtype=float)).sum()
    total_sac  = results.get('Saccardi_flux_TgC_yr',      pd.Series(dtype=float)).sum()
    total_gw   = results.get('GW_CO2_flux_TgC_yr',        pd.Series(dtype=float)).sum()
    total_melt = results.get('ice_melt_emission_TgC_yr',  pd.Series(dtype=float)).sum()
    total_huc_area = results['huc2_area_m2'].sum()

    # Liu CONUS bounds (1σ)
    total_liu_sigma = results['Liu_flux_sigma_TgC_yr'].sum()
    total_liu_lo = total_fin - total_liu_sigma
    total_liu_hi = total_fin + total_liu_sigma

    # GW CONUS bounds (90% CI)
    if not np.isnan(conus_sigma_huc4_gw):
        total_gw_lo = max(0, total_gw - Z_90 * conus_sigma_huc4_gw)
        total_gw_hi = total_gw + Z_90 * conus_sigma_huc4_gw
    elif not np.isnan(conus_sigma_huc2_gw):
        total_gw_lo = max(0, total_gw - Z_90 * conus_sigma_huc2_gw)
        total_gw_hi = total_gw + Z_90 * conus_sigma_huc2_gw
    else:
        total_gw_lo = np.nan
        total_gw_hi = np.nan

    total_b    = results.get('Butman_stream_central_TgC', pd.Series(dtype=float)).sum()
    total_b_lo = results.get('Butman_stream_lower_TgC',   pd.Series(dtype=float)).sum()
    total_b_hi = results.get('Butman_stream_upper_TgC',   pd.Series(dtype=float)).sum()

    pct_ce = total_gw    / total_fin    * 100 if total_fin    > 0 else np.nan
    pct_lo = total_gw_lo / total_liu_hi * 100 if total_liu_hi > 0 else np.nan
    pct_hi = total_gw_hi / total_liu_lo * 100 if total_liu_lo > 0 else np.nan

    print(
        f"{'TOT':>4}  "
        f"{total_liu_lo:>10.3f} {total_fin:>10.3f} {total_liu_hi:>10.3f}  "
        f"{total_gw_lo:>10.4f} {total_gw:>10.4f} {total_gw_hi:>10.4f}  "
        f"{pct_lo:>7.1f}% {pct_ce:>7.1f}% {pct_hi:>7.1f}%"
        f"   Liu=1σ, GW=90%CI (HUC4-block)"
    )

    #  Summary
    print(f"\n  CORRECTION CASCADE:")
    print(f"  {'Liu raw (no correction):':<50} {total_raw:>8.2f} TgC/yr")
    print(f"  {'Liu ephemeral corrected (Ray 2013):':<50} {total_eph:>8.2f} TgC/yr  "
          f"({(1-total_eph/total_raw)*100:>5.1f}% reduction)")
    print(f"  {'Liu ephemeral + ice corrected:':<50} {total_fin - total_melt:>8.2f} TgC/yr  "
          f"({(1-(total_fin-total_melt)/total_eph)*100:>5.1f}% further reduction)")
    print(f"  {'Ice-melt emissions added back:':<50} +{total_melt:>7.4f} TgC/yr")
    print(f"  {'Liu FINAL:':<50} {total_fin:>8.2f} ± {total_liu_sigma:.2f} TgC/yr (1σ)")

    print(f"\n  COMPARISON:")
    print(f"  {'Liu (this calc, 1σ):':<50} {total_fin:>8.2f} "
          f"[{total_liu_lo:.2f}, {total_liu_hi:.2f}] TgC/yr")
    print(f"  {'Liu (their US estimate, Table S5):':<50} {'57':>8} TgC/yr")
    print(f"  {'Saccardi (transport model):':<50} {total_sac:>8.2f} "
          f"± {SACCARDI_PUBLISHED_DF:.0f} TgC/yr")
    print(f"  {'Butman (statistical upscaling, 90% CI):':<50} {total_b:>8.1f} "
          f"[{total_b_lo:.1f}, {total_b_hi:.1f}] TgC/yr")
    print(f"  {'GW CO2 flux (this study, 90% CI):':<50} {total_gw:>8.4f} "
          f"[{total_gw_lo:.4f}, {total_gw_hi:.4f}] TgC/yr  "
          f"(HUC4-block σ={conus_sigma_huc4_gw:.4f})")

    #  GW as % of Liu
    print(f"\n  GW as % of Liu final (Liu: 1σ; GW: 90% CI):")
    print(f"    Central:   {pct_ce:.1f}%")
    print(f"    Bounds:    [{pct_lo:.1f}%, {pct_hi:.1f}%]  (GW_lo/Liu_hi, GW_hi/Liu_lo)")
    pct_lo_gw = total_gw_lo / total_fin * 100
    pct_hi_gw = total_gw_hi / total_fin * 100
    print(f"    GW only:   [{pct_lo_gw:.1f}%, {pct_hi_gw:.1f}%]  (Liu fixed)")

    #  GW as % of Saccardi
    if total_sac > 0:
        pct_sac_ce = total_gw / total_sac * 100
        b_lo_sac = total_gw_lo / (total_sac + SACCARDI_PUBLISHED_DF) * 100
        b_hi_sac = total_gw_hi / max(total_sac - SACCARDI_PUBLISHED_DF, 1e-9) * 100
        pct_lo_gw_s = total_gw_lo / total_sac * 100
        pct_hi_gw_s = total_gw_hi / total_sac * 100
        print(f"\n  GW as % of Saccardi (Saccardi: published ±{SACCARDI_PUBLISHED_DF:.0f}; GW: 90% CI):")
        print(f"    Central:   {pct_sac_ce:.1f}%")
        print(f"    Bounds:    [{b_lo_sac:.1f}%, {b_hi_sac:.1f}%]")
        print(f"    GW only:   [{pct_lo_gw_s:.1f}%, {pct_hi_gw_s:.1f}%]  (Saccardi fixed)")

    #  GW as % of Butman
    if total_b > 0:
        b_pct_ce = total_gw / total_b * 100
        b_pct_lo = total_gw_lo / total_b_hi * 100 if total_b_hi > 0 else np.nan
        b_pct_hi = total_gw_hi / total_b_lo * 100 if total_b_lo > 0 else np.nan
        pct_lo_gw_b = total_gw_lo / total_b * 100
        pct_hi_gw_b = total_gw_hi / total_b * 100
        print(f"\n  GW as % of Butman (90% CI):")
        print(f"    Central:   {b_pct_ce:.1f}%")
        print(f"    Bounds:    [{b_pct_lo:.1f}%, {b_pct_hi:.1f}%]")
        print(f"    GW only:   [{pct_lo_gw_b:.1f}%, {pct_hi_gw_b:.1f}%]  (Butman fixed)")

    #  Yields
    print(f"\n  YIELDS (gC/m²/yr normalized by HUC2 watershed area):")
    print(f"  {'Liu final:':<45} {total_fin / TG_PER_G / total_huc_area:.4f}")
    print(f"  {'Saccardi:':<45} {total_sac / TG_PER_G / total_huc_area:.4f}")
    if total_b > 0:
        print(f"  {'Butman:':<45} {total_b / TG_PER_G / total_huc_area:.4f}")
    print(f"  {'GW:':<45} {total_gw / TG_PER_G / total_huc_area:.4f}")

    results.to_csv(OUTPUT_PATH)
    print(f"\nSaved: {OUTPUT_PATH}")
    return results


if __name__ == '__main__':
    results = main()
