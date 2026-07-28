"""
figures_si.py

Supporting-information and contribution figures:
  Figure 3   HUC2 groundwater contribution choropleth (main text)
  Figure S4  HUC2 contribution barplot across the three efflux datasets
  Figure S2b ERA5-Land subsurface runoff map

Requires the *_stream_co2_by_huc2_*.csv outputs from the contribution scripts
and the ERA5-Land runoff NetCDFs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "figure.titlesize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

BASE_DIR = Path("data")
FIG_DIR  = Path("figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)
HUC_PATH = BASE_DIR / "shp_files/Watershed_Boundary_Dataset_HUC_2.gpkg"

co2_predictions = "co2_predictions_aq_uncertainty2"
tag = "aq" if co2_predictions.endswith("_aq_uncertainty2") else "ppm"


def plot_huc2_contribution_choropleth():

    # ============================================================
    # STYLE — paper/presentation quality
    # ============================================================

    # ============================================================
    # DATA — load dynamically from results CSVs
    # ============================================================

    # Determine tag

    # Load results CSV (output of liu_stream_co2_by_huc2.py)
    liu_csv = pd.read_csv(
        BASE_DIR / f'liu_stream_co2_by_huc2_{tag}.csv',
        index_col='HUC', dtype={'HUC': str}
    )
    #print(liu_csv.columns.tolist())

    hucs = gpd.read_file(HUC_PATH)
    CONUS_HUC2 = [str(i).zfill(2) for i in range(1, 19)]
    hucs = hucs[hucs['huc2'].isin(CONUS_HUC2)].to_crs('EPSG:4326')

    #  Build per-HUC data from CSV columns
    data = pd.DataFrame({'HUC': CONUS_HUC2}).set_index('HUC')

    # GW flux
    data['GW_flux_TgC'] = liu_csv['GW_CO2_flux_TgC_yr']

    # GW % of each method (central + bounds)
    for method, col_central, col_lower, col_upper in [
        ('Butman',   'GW_pct_of_Butman',    'GW_pct_of_Butman_lower',  'GW_pct_of_Butman_upper'),
        ('Liu',      'GW_pct_of_Liu_final', 'GW_pct_of_Liu_lower',     'GW_pct_of_Liu_upper'),
        ('Saccardi', 'GW_pct_of_Saccardi',  'GW_pct_of_Saccardi_lower', 'GW_pct_of_Saccardi_upper'),
    ]:
        if col_central in liu_csv.columns:
            data[f'GW_pct_{method}'] = liu_csv[col_central]
        if col_lower and col_lower in liu_csv.columns:
            data[f'GW_pct_{method}_lower'] = liu_csv[col_lower]
        else:
            data[f'GW_pct_{method}_lower'] = np.nan
        if col_upper and col_upper in liu_csv.columns:
            data[f'GW_pct_{method}_upper'] = liu_csv[col_upper]
        else:
            data[f'GW_pct_{method}_upper'] = np.nan

    # Average across methods
    pct_cols = [c for c in ['GW_pct_Butman', 'GW_pct_Liu', 'GW_pct_Saccardi'] if c in data.columns]
    data['GW_pct_Average'] = data[pct_cols].mean(axis=1)

    lo_cols = [c for c in ['GW_pct_Butman_lower', 'GW_pct_Liu_lower', 'GW_pct_Saccardi_lower'] if c in data.columns]
    hi_cols = [c for c in ['GW_pct_Butman_upper', 'GW_pct_Liu_upper', 'GW_pct_Saccardi_upper'] if c in data.columns]
    data['GW_pct_Average_lower'] = data[lo_cols].mean(axis=1)
    data['GW_pct_Average_upper'] = data[hi_cols].mean(axis=1)

    # CONUS totals for subtitles
    total_gw = data['GW_flux_TgC'].sum()
    total_liu = liu_csv['Liu_flux_final_TgC_yr'].sum() if 'Liu_flux_final_TgC_yr' in liu_csv.columns else np.nan
    total_butman = liu_csv['Butman_stream_central_TgC'].sum() if 'Butman_stream_central_TgC' in liu_csv.columns else np.nan
    total_saccardi = liu_csv['Saccardi_flux_TgC_yr'].sum() if 'Saccardi_flux_TgC_yr' in liu_csv.columns else np.nan

    stream_totals = [t for t in [total_butman, total_liu, total_saccardi] if pd.notna(t)]
    mean_stream_total = np.mean(stream_totals) if stream_totals else np.nan
    mean_pct = total_gw / mean_stream_total * 100 if pd.notna(mean_stream_total) and mean_stream_total > 0 else np.nan

    # Merge into geodataframe
    data = data.reset_index()
    hucs = hucs.merge(data, left_on='huc2', right_on='HUC', how='inner')

    # ============================================================
    # SHARED SETTINGS
    # ============================================================
    methods = {
        'GW_pct_Butman':   ('Groundwater as Percent of Butman et al. (2016) Stream CO₂ Efflux',
                            f'Butman et al. Total: {total_butman:.1f} TgC/yr',
                            f'GW: {total_gw:.2f} TgC/yr ({total_gw/total_butman*100:.1f}%)' if pd.notna(total_butman) else ''),
        'GW_pct_Liu':      ('Groundwater as Percent of Liu et al. (2022) Stream CO₂ Efflux',
                            f'Liu et al. Total: {total_liu:.2f} TgC/yr',
                            f'GW: {total_gw:.2f} TgC/yr ({total_gw/total_liu*100:.1f}%)' if pd.notna(total_liu) else ''),
        'GW_pct_Saccardi': ('Groundwater as Percent of Saccardi et al. (2024) Stream CO₂ Efflux',
                            f'Saccardi et al. Total: {total_saccardi:.2f} TgC/yr',
                            f'GW: {total_gw:.2f} TgC/yr ({total_gw/total_saccardi*100:.1f}%)' if pd.notna(total_saccardi) else ''),
        'GW_pct_Average':  ('Groundwater as Percent of Mean Stream CO₂ Efflux',
                            f'Mean Stream Total: {mean_stream_total:.1f} TgC/yr',
                            f'GW: {total_gw:.2f} TgC/yr ({mean_pct:.1f}%)'),
    }

    vmin, vmax = 0, 50
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=vmin, vmax=40)

    #  Label positions: (dx, dy) offsets from centroid
    nudge = {
        '01': (3.5,  1.5),
        '02': (2.5, -1.0),
        '03': (0.0, -2.5),   # nudge FL/SE down
        '04': (-1.5, 1.0),
        '06': (2.0, -1.5),
        '09': (0.0,  1.0),
        '15': (1.5, -1.0),
        '16': (0.0, -0.3),
        '18': (2.5, -1.5),
    }

    # ============================================================
    # GENERATE 4 SEPARATE FIGURES
    # ============================================================
    for col, (title, line1, line2) in methods.items():

        col_lower = f'{col}_lower'
        col_upper = f'{col}_upper'

        fig, ax = plt.subplots(figsize=(44, 34), dpi=350)

        # Map
        hucs.plot(
            column=col,
            cmap=cmap,
            norm=norm,
            edgecolor='black',
            linewidth=1.0,
            ax=ax,
            legend=False,
        )

        # Annotate HUC2s
        for _, row in hucs.iterrows():
            centroid = row.geometry.centroid
            pct = row[col]
            flux = row['GW_flux_TgC']
            huc_id = row['huc2']
            pct_lo = row.get(col_lower, np.nan) if col_lower in hucs.columns else np.nan
            pct_hi = row.get(col_upper, np.nan) if col_upper in hucs.columns else np.nan

            cx, cy = centroid.x, centroid.y
            x, y = cx, cy

            if huc_id in nudge:
                dx, dy = nudge[huc_id]
                x += dx
                y += dy

            # Build annotation text — HUC ID + % first, uncertainty second, TgC third
            if pd.notna(pct_lo) and pd.notna(pct_hi) and pct_lo > 0:
                hw_lo = pct - pct_lo
                hw_hi = pct_hi - pct
                if abs(hw_lo - hw_hi) < 1.0:
                    hw = (hw_lo + hw_hi) / 2
                    pct_line  = f'{huc_id}: {pct:.1f}%'
                    rest_line = f'±{hw:.1f}%\n({flux:.2f} TgC)'
                else:
                    pct_line  = f'{huc_id}: {pct:.1f}%'
                    rest_line = f'[{pct_lo:.1f}–{pct_hi:.1f}%]\n({flux:.2f} TgC)'
            else:
                pct_line  = f'{huc_id}: {pct:.1f}%'
                rest_line = f'\n({flux:.2f} TgC)'

            # 1) Invisible bbox sized for full text (provides white background)
            ax.annotate(
                pct_line + '\n' + rest_line,
                xy=(x, y),
                fontsize=35,
                color='none',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='gray', alpha=0.85),
            )
            # 2) Bold HUC ID + % line on top
            ax.annotate(
                pct_line,
                xy=(x, y + 0.9),
                fontsize=36,
                fontweight='bold',
                color='black',
                ha='center', va='center',
            )
            # 3) Normal weight uncertainty + flux below
            ax.annotate(
                rest_line,
                xy=(x, y - 0.5),
                fontsize=34,
                fontweight='normal',
                color='black',
                ha='center', va='center',
            )

        ax.set_title(title, fontsize=35, fontweight='bold', pad=12)
        ax.set_xlim(-130, -62)
        ax.set_ylim(22, 52)
        ax.set_aspect('equal')
        ax.axis('off')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.7)
        cbar.set_label('GW CO₂ Contribution (%)', fontsize=44, labelpad=14)
        cbar.ax.tick_params(labelsize=38)
        cbar.ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

        #  CONUS summary box
        gw_lo  = liu_csv['GW_CO2_flux_lower_TgC_yr'].sum()
        gw_hi  = liu_csv['GW_CO2_flux_upper_TgC_yr'].sum()

        liu_c  = total_gw / total_liu  * 100
        liu_lo = gw_lo / liu_csv['Liu_flux_upper_TgC_yr'].sum() * 100
        liu_hi = gw_hi / liu_csv['Liu_flux_lower_TgC_yr'].sum() * 100

        sac_c  = total_gw / total_saccardi * 100
        sac_lo = gw_lo / (total_saccardi + 23) * 100
        sac_hi = gw_hi / (total_saccardi - 23) * 100

        but_c  = total_gw / total_butman * 100
        but_lo = gw_lo / liu_csv['Butman_stream_upper_TgC'].sum() * 100
        but_hi = gw_hi / liu_csv['Butman_stream_lower_TgC'].sum() * 100

        avg_c  = np.mean([liu_c,  sac_c,  but_c])
        avg_lo = min(liu_lo, sac_lo, but_lo)
        avg_hi = max(liu_hi, sac_hi, but_hi)

        gw_line = f"CONUS GW CO₂ = {total_gw:.2f} TgC/yr  [{gw_lo:.2f}–{gw_hi:.2f}]\n"
        div     = f"{''*46}\n"

        if col == 'GW_pct_Liu':
            summary = (
                gw_line + div +
                f"GW Contribution to Liu et al. (2022)       {liu_c:5.1f}%  [{liu_lo:.1f}–{liu_hi:.1f}%]"
            )
        elif col == 'GW_pct_Saccardi':
            summary = (
                gw_line + div +
                f"GW Contribution to Saccardi et al. (2024)  {sac_c:5.1f}%  [{sac_lo:.1f}–{sac_hi:.1f}%]"
            )
        elif col == 'GW_pct_Butman':
            summary = (
                gw_line + div +
                f"GW Contribution to Butman et al. (2016)    {but_c:5.1f}%  [{but_lo:.1f}–{but_hi:.1f}%]"
            )
        elif col == 'GW_pct_Average':
            summary = (
                gw_line + div +
                # f"GW Contribution to Liu et al. (2022)       {liu_c:5.1f}%  [{liu_lo:.1f}–{liu_hi:.1f}%]\n"
                # f"GW Contribution to Saccardi et al. (2024)  {sac_c:5.1f}%  [{sac_lo:.1f}–{sac_hi:.1f}%]\n"
                # f"GW Contribution to Butman et al. (2016)    {but_c:5.1f}%  [{but_lo:.1f}–{but_hi:.1f}%]\n" +
                # div +
                f"Mean GW Contribution          {avg_c:5.1f}%  [{avg_lo:.1f}–{avg_hi:.1f}%]"
            )

        ax.text(
            0.01, 0.005, summary,
            transform=ax.transAxes,
            fontsize=30, fontfamily='serif',
            va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                      edgecolor='#333333', linewidth=1.8, alpha=0.93),
            zorder=10,
        )

        # Save
        safe_name = col.lower().replace('gw_pct_', '')
        out_path = FIG_DIR / f'gw_co2_contribution_huc2_{safe_name}.png'
        fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"Saved: {out_path}")
        plt.close(fig)

    print("\nAll 4 figures saved.")

def plot_huc2_contribution_barplot():

    # ============================================================
    # STYLE
    # ============================================================

    # ============================================================
    # LOAD DATA
    # ============================================================


    Z_90 = 1.645

    # Primary: Liu CSV (Liu bounds + GW % of Liu + Saccardi)
    liu_path = BASE_DIR / f'liu_stream_co2_by_huc2_{tag}.csv'
    df = pd.read_csv(liu_path, index_col='HUC', dtype={'HUC': str})
    print(f"Loaded Liu CSV: {liu_path}")

    # Saccardi CSV (for Saccardi bounds if present)
    sac_path = BASE_DIR / f'saccardi_stream_co2_by_huc2_{tag}.csv'
    sac = pd.read_csv(sac_path, index_col='HUC', dtype={'HUC': str}) if sac_path.exists() else None
    if sac is not None:
        print(f"Loaded Saccardi CSV: {sac_path}")

    # GW flux CSV (for sigma columns + Butman bounds)
    gw_path = BASE_DIR / f'gw_co2_flux_by_huc2_1990_2010_{tag}.csv'
    gw = pd.read_csv(gw_path, index_col='HUC', dtype={'HUC': str})
    print(f"Loaded GW CSV: {gw_path}")
    print(f"  GW columns: {gw.columns.tolist()}")

    # ============================================================
    # COMPUTE CONUS-LEVEL GW BOUNDS FROM BLOCK SIGMA
    # Prefer HUC4-block σ (primary); fall back to HUC2-block σ
    # ============================================================
    if 'GW_CO2_flux_sigma_huc4_TgC_yr' in gw.columns:
        sigma_col = 'GW_CO2_flux_sigma_huc4_TgC_yr'
        sigma_label = 'HUC4-block'
    elif 'GW_CO2_flux_sigma_huc2_TgC_yr' in gw.columns:
        sigma_col = 'GW_CO2_flux_sigma_huc2_TgC_yr'
        sigma_label = 'HUC2-block'
    elif 'GW_CO2_flux_sigma_TgC_yr' in gw.columns:
        sigma_col = 'GW_CO2_flux_sigma_TgC_yr'
        sigma_label = 'HUC2-block (legacy)'
    else:
        sigma_col = None
        sigma_label = None

    if sigma_col is not None:
        per_huc_sigma = gw[sigma_col].dropna()
        conus_sigma_gw = float(np.sqrt((per_huc_sigma**2).sum()))
        total_gw_central = gw['GW_CO2_flux_TgC_yr'].sum()
        total_gw_lo = max(0.0, total_gw_central - Z_90 * conus_sigma_gw)
        total_gw_hi = total_gw_central + Z_90 * conus_sigma_gw

        # Scale per-HUC bounds proportionally from CONUS block CI.
        # This preserves the relative spatial pattern while using the
        # physically motivated block aggregation for the total uncertainty.
        scale_lo = total_gw_lo / total_gw_central
        scale_hi = total_gw_hi / total_gw_central
        gw['GW_CO2_flux_lower_TgC_yr'] = gw['GW_CO2_flux_TgC_yr'] * scale_lo
        gw['GW_CO2_flux_upper_TgC_yr'] = gw['GW_CO2_flux_TgC_yr'] * scale_hi

        print(f"\n  GW σ method: {sigma_label}")
        print(f"  CONUS σ:     {conus_sigma_gw:.4f} TgC/yr ({conus_sigma_gw/total_gw_central*100:.1f}% rel)")
        print(f"  CONUS 90% CI: [{total_gw_lo:.4f}, {total_gw_hi:.4f}] TgC/yr")
        print(f"  Per-HUC scale factors: lo={scale_lo:.4f}, hi={scale_hi:.4f}")
    else:
        # Last resort: use existing lower/upper columns if present
        if 'GW_CO2_flux_lower_TgC_yr' in gw.columns:
            print("   No sigma column found — using existing lower/upper columns")
            total_gw_central = gw['GW_CO2_flux_TgC_yr'].sum()
            total_gw_lo = gw['GW_CO2_flux_lower_TgC_yr'].sum()
            total_gw_hi = gw['GW_CO2_flux_upper_TgC_yr'].sum()
            sigma_label = 'legacy bounds'
        else:
            print("   No σ or bounds found — uncertainty will be NaN")
            gw['GW_CO2_flux_lower_TgC_yr'] = np.nan
            gw['GW_CO2_flux_upper_TgC_yr'] = np.nan
            total_gw_central = gw['GW_CO2_flux_TgC_yr'].sum()
            total_gw_lo = np.nan
            total_gw_hi = np.nan
            sigma_label = 'unavailable'

    # ============================================================
    # BUILD PER-HUC BOUNDS FOR ALL THREE METHODS
    # ============================================================

    #  Butman bounds
    # Prefer precomputed columns in GW CSV; fall back to computing from flux bounds
    if 'GW_pct_of_Butman_lower' in df.columns and 'GW_pct_of_Butman_upper' in df.columns:
        print("   Butman bounds already in Liu CSV")
    elif 'Butman_stream_lower_TgC' in df.columns:
        df['GW_pct_of_Butman_lower'] = (
            gw['GW_CO2_flux_lower_TgC_yr'] / df['Butman_stream_upper_TgC'] * 100
        )
        df['GW_pct_of_Butman_upper'] = (
            gw['GW_CO2_flux_upper_TgC_yr'] / df['Butman_stream_lower_TgC'] * 100
        )
        print("   Butman bounds computed from flux bounds")
    else:
        df['GW_pct_of_Butman_lower'] = np.nan
        df['GW_pct_of_Butman_upper'] = np.nan
        print("   Butman bounds unavailable")

    #  Saccardi bounds
    if 'GW_pct_of_Saccardi_lower' in df.columns and 'GW_pct_of_Saccardi_upper' in df.columns:
        print("   Saccardi bounds already in Liu CSV")
    elif sac is not None and 'GW_pct_of_Saccardi_lower' in sac.columns:
        df['GW_pct_of_Saccardi_lower'] = sac['GW_pct_of_Saccardi_lower']
        df['GW_pct_of_Saccardi_upper'] = sac['GW_pct_of_Saccardi_upper']
        print("   Saccardi bounds from Saccardi CSV")
    elif sac is not None and 'Saccardi_flux_upper_TgC_yr' in sac.columns:
        df['GW_pct_of_Saccardi_lower'] = (
            gw['GW_CO2_flux_lower_TgC_yr'] / sac['Saccardi_flux_upper_TgC_yr'] * 100
        )
        df['GW_pct_of_Saccardi_upper'] = (
            gw['GW_CO2_flux_upper_TgC_yr'] / sac['Saccardi_flux_lower_TgC_yr'] * 100
        )
        print("   Saccardi bounds computed from flux bounds")
    elif 'Saccardi_flux_TgC_yr' in df.columns:
        df['GW_pct_of_Saccardi_lower'] = (
            gw['GW_CO2_flux_lower_TgC_yr'] / df['Saccardi_flux_TgC_yr'] * 100
        )
        df['GW_pct_of_Saccardi_upper'] = (
            gw['GW_CO2_flux_upper_TgC_yr'] / df['Saccardi_flux_TgC_yr'] * 100
        )
        print("   Saccardi bounds computed (GW uncertainty only, Saccardi fixed)")
    else:
        df['GW_pct_of_Saccardi_lower'] = np.nan
        df['GW_pct_of_Saccardi_upper'] = np.nan
        print("   Saccardi bounds unavailable")

    #  Liu bounds
    if 'GW_pct_of_Liu_lower' in df.columns:
        print("   Liu bounds already in Liu CSV")
    else:
        df['GW_pct_of_Liu_lower'] = (
            gw['GW_CO2_flux_lower_TgC_yr'] / df['Liu_flux_upper_TgC_yr'] * 100
        )
        df['GW_pct_of_Liu_upper'] = (
            gw['GW_CO2_flux_upper_TgC_yr'] / df['Liu_flux_lower_TgC_yr'] * 100
        )
        print("   Liu bounds computed from flux bounds")

    # Diagnostic
    print(f"\n  Per-HUC bound ranges:")
    for col in ['GW_pct_of_Butman', 'GW_pct_of_Butman_lower', 'GW_pct_of_Butman_upper',
                'GW_pct_of_Liu_final', 'GW_pct_of_Liu_lower', 'GW_pct_of_Liu_upper',
                'GW_pct_of_Saccardi', 'GW_pct_of_Saccardi_lower', 'GW_pct_of_Saccardi_upper']:
        if col in df.columns and df[col].notna().any():
            print(f"    {col:<35}: {df[col].min():.1f} – {df[col].max():.1f}")

    # ============================================================
    # CONUS TOTALS FOR ANNOTATION BOX
    # ============================================================
    total_gw    = df['GW_CO2_flux_TgC_yr'].sum()
    total_liu   = df['Liu_flux_final_TgC_yr'].sum()   if 'Liu_flux_final_TgC_yr'     in df.columns else np.nan
    total_sac   = df['Saccardi_flux_TgC_yr'].sum()    if 'Saccardi_flux_TgC_yr'      in df.columns else np.nan
    total_but   = df['Butman_stream_central_TgC'].sum() if 'Butman_stream_central_TgC' in df.columns else np.nan

    # ============================================================
    # PLOT SETUP
    # ============================================================
    hucs  = df.index.tolist()
    n     = len(hucs)
    x     = np.arange(n)
    width = 0.25

    bar_specs = {
        'Butman': {
            'central': 'GW_pct_of_Butman',
            'lower':   'GW_pct_of_Butman_lower',
            'upper':   'GW_pct_of_Butman_upper',
            'color':   'cornflowerblue',
            'label':   'Butman et al. (2016)',
        },
        'Liu': {
            'central': 'GW_pct_of_Liu_final',
            'lower':   'GW_pct_of_Liu_lower',
            'upper':   'GW_pct_of_Liu_upper',
            'color':   '#1b9e77',
            'label':   'Liu et al. (2022)',
        },
        'Saccardi': {
            'central': 'GW_pct_of_Saccardi',
            'lower':   'GW_pct_of_Saccardi_lower',
            'upper':   'GW_pct_of_Saccardi_upper',
            'color':   '#d95f02',
            'label':   'Saccardi et al. (2024)',
        },
    }

    # ============================================================
    # PLOT
    # ============================================================
    fig, ax = plt.subplots(figsize=(18, 8))

    for i, (key, spec) in enumerate(bar_specs.items()):
        vals   = df[spec['central']].fillna(0).values
        lo_col = spec['lower']
        hi_col = spec['upper']

        lo_vals = df[lo_col].fillna(0).values if lo_col in df.columns else np.zeros(n)
        hi_vals = df[hi_col].fillna(0).values if hi_col in df.columns else np.zeros(n)

        err_lo = np.clip(vals - lo_vals, 0, None)
        err_hi = np.clip(hi_vals - vals, 0, None)
        yerr   = np.array([err_lo, err_hi])

        bars = ax.bar(
            x + i * width,
            vals,
            width,
            yerr=yerr,
            capsize=2,
            error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor='dimgray', alpha=0.7),
            label=spec['label'],
            color=spec['color'],
            edgecolor='white',
            linewidth=0.5,
            zorder=3,
        )

        # Value labels above each bar
        for j, (bar, val, hi) in enumerate(zip(bars, vals, hi_vals)):
            if val > 1.5:
                y_pos = min(hi, val + err_hi[j]) + 0.8
                y_pos = min(y_pos, 98)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f'{val:.1f}',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                    color=spec['color'],
                    clip_on=True,
                )

    #  Axes
    ax.set_xlabel('HUC2', fontsize=22, fontweight='bold')
    ax.set_ylabel('GW CO₂ as Percent of Stream Efflux', fontsize=22, fontweight='bold')
    ax.set_title('Groundwater CO₂ Contribution to Stream Efflux by HUC2',
                 fontsize=22, fontweight='bold', pad=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(hucs, fontsize=18)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    #  Legend
    ax.legend(
        fontsize=16,
        frameon=True,
        framealpha=0.9,
        edgecolor='gray',
        loc='upper right',
        title='Stream Efflux Estimate',
        title_fontsize=16,
    )

    #  CONUS totals annotation box
    gw_lo_str = f'{total_gw_lo:.2f}' if not np.isnan(total_gw_lo) else 'N/A'
    gw_hi_str = f'{total_gw_hi:.2f}' if not np.isnan(total_gw_hi) else 'N/A'

    summary_lines = [
        f'CONUS Totals (90% CI):',
        f'  GW:       {total_gw:.2f} [{gw_lo_str}, {gw_hi_str}] TgC/yr',
    ]
    if not np.isnan(total_but):
        summary_lines.append(f'  Butman:   {total_but:.1f} TgC/yr (GW={total_gw/total_but*100:.1f}%)')
    if not np.isnan(total_liu):
        summary_lines.append(f'  Liu:      {total_liu:.2f} TgC/yr (GW={total_gw/total_liu*100:.1f}%)')
    if not np.isnan(total_sac):
        summary_lines.append(f'  Saccardi: {total_sac:.1f} TgC/yr  (GW={total_gw/total_sac*100:.1f}%)')

    ax.text(
        0.98, 0.72,
        '\n'.join(summary_lines),
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=15,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                  edgecolor='gray', alpha=0.9),
    )

    #  Clipped bars note
    all_uppers = np.concatenate([
        df[spec['upper']].fillna(0).values
        for spec in bar_specs.values()
        if spec['upper'] in df.columns
    ])
    if (all_uppers > 100).any():
        ax.text(
            0.02, 0.97,
            'Note: Some upper error bars extend beyond axis limit',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=15, fontstyle='italic', color='gray',
        )

    plt.tight_layout()

    out_path = FIG_DIR / f'gw_co2_pct_barplot_by_huc2_{tag}.png'
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {out_path}")
    plt.close(fig)

# --- Figure S2b: ERA5-Land subsurface runoff map ---
#!/usr/bin/env python3
"""
map_ssro.py
===========
CONUS map of mean annual sub-surface runoff (SSRO) from ERA5-Land,
clipped to US state boundaries. Only US states drawn — no foreign
borders, no rivers, no lakes.

Usage
-----
    python map_ssro.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import warnings

warnings.filterwarnings('ignore')

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from shapely.geometry import box as shapely_box

# ============================================================================
# CONFIG
# ============================================================================
SSRO_PATH = "data/runoff/era5_land_mean_annual_runoff_1979_2014_nofreeze.nc"
SAVE_PATH = "figs/figS2b_ssro_mean_annual_map.png"

CONUS_EXTENT = [-125, -66, 24, 50]

CONVERT_TO_MM = True


# ============================================================================
# US STATE CLIP PATH
# ============================================================================
_CLIP_CACHE = {}
_US_GEOMS_CACHE = None


def _get_us_conus_geoms():
    """Load US CONUS state geometries once, cache for reuse."""
    global _US_GEOMS_CACHE
    if _US_GEOMS_CACHE is not None:
        return _US_GEOMS_CACHE

    shpfile = shpreader.natural_earth(
        resolution='50m', category='cultural',
        name='admin_1_states_provinces_lakes')
    reader = shpreader.Reader(shpfile)

    conus_box = shapely_box(CONUS_EXTENT[0], CONUS_EXTENT[2],
                            CONUS_EXTENT[1], CONUS_EXTENT[3])

    geoms = []
    for rec in reader.records():
        if rec.attributes.get('admin') != 'United States of America':
            continue
        g = rec.geometry
        if not g.centroid.within(conus_box):
            continue
        clipped = g.intersection(conus_box)
        if not clipped.is_empty:
            geoms.append(clipped)

    _US_GEOMS_CACHE = geoms
    print(f"  [clip] {len(geoms)} CONUS state geometries loaded")
    return _US_GEOMS_CACHE


def _get_us_clip_patch(ax):
    proj = ax.projection
    proj_key = str(type(proj))

    if proj_key not in _CLIP_CACHE:
        print("  [clip] Building US state boundary clip path...")
        geoms = _get_us_conus_geoms()
        if not geoms:
            return None

        us_conus = unary_union(geoms)
        projected = proj.project_geometry(us_conus, ccrs.PlateCarree())

        all_verts, all_codes = [], []
        parts = list(projected.geoms) if hasattr(projected, 'geoms') else [projected]
        for geom in parts:
            if not hasattr(geom, 'exterior'):
                continue
            ring = np.array(geom.exterior.coords)
            n = len(ring)
            if n < 3:
                continue
            all_verts.extend(ring.tolist())
            all_codes.extend(
                [MplPath.MOVETO] +
                [MplPath.LINETO] * (n - 2) +
                [MplPath.CLOSEPOLY])
            for interior in geom.interiors:
                hole = np.array(interior.coords)
                nh = len(hole)
                if nh < 3:
                    continue
                all_verts.extend(hole.tolist())
                all_codes.extend(
                    [MplPath.MOVETO] +
                    [MplPath.LINETO] * (nh - 2) +
                    [MplPath.CLOSEPOLY])

        if not all_verts:
            return None
        _CLIP_CACHE[proj_key] = MplPath(all_verts, all_codes)
        print("  [clip] Done — cached")

    patch = mpatches.PathPatch(
        _CLIP_CACHE[proj_key],
        transform=ax.transData,
        facecolor='none', edgecolor='none')
    ax.add_patch(patch)
    return patch


# ============================================================================
# COLORMAP
# ============================================================================
def get_ssro_cmap():
    colors = [
        '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
        '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b',
    ]
    return mcolors.LinearSegmentedColormap.from_list('ssro_blues', colors, N=256)


# ============================================================================
# MAIN
# ============================================================================
def map_ssro(nc_path=SSRO_PATH, save_path=SAVE_PATH):
    #  Load data
    print(f"Loading: {nc_path}")
    ds = xr.open_dataset(nc_path)

    lats = ds['latitude'].values
    lons = ds['longitude'].values
    ssro = ds['ssro'].values

    unit_label = 'm yr⁻¹'
    if CONVERT_TO_MM:
        ssro = ssro * 1000.0
        unit_label = 'mm yr⁻¹'

    ssro = np.where(ssro <= 0, np.nan, ssro)

    print(f"  Grid: {len(lats)} lat × {len(lons)} lon")
    valid = ssro[~np.isnan(ssro)]
    print(f"  SSRO stats ({unit_label}): "
          f"mean={np.mean(valid):.1f}, median={np.median(valid):.1f}, "
          f"max={np.max(valid):.1f}")

    #  Color limits
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)

    cmap = get_ssro_cmap().copy()
    cmap.set_bad(color='none')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    #  Figure
    proj = ccrs.AlbersEqualArea(
        central_longitude=-96, central_latitude=37.5,
        standard_parallels=(29.5, 45.5))

    fig, ax = plt.subplots(figsize=(14, 8),
                           subplot_kw={'projection': proj})
    ax.set_extent(CONUS_EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor('white')

    #  US state boundaries ONLY (no foreign borders, no lakes)
    us_geoms = _get_us_conus_geoms()
    for g in us_geoms:
        ax.add_geometries([g], ccrs.PlateCarree(),
                          facecolor='none', edgecolor='#444444',
                          linewidth=0.5, zorder=3)

    #  Plot
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    im = ax.pcolormesh(
        lon_grid, lat_grid, ssro,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        shading='auto', rasterized=True, zorder=2)

    clip_patch = _get_us_clip_patch(ax)
    if clip_patch is not None:
        im.set_clip_path(clip_patch)

    #  Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label(f'Sub-Surface Runoff ({unit_label})',
                   fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    #  Title
    ax.set_title(
        'Mean Annual Sub-Surface Runoff (ERA5-Land, 1970–2000)',
        fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig



if __name__ == "__main__":
    plot_huc2_contribution_choropleth()
    plot_huc2_contribution_barplot()
    map_ssro()
