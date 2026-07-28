"""
figures_main.py

Main-text figures built directly from the cleaned training table:
  Figure 1  well sampling map (a) + aridity boxplots by HUC2 (b)
  Figure S1 landscape/climate driver panels + geology/land-use panels

Run:  python figures_main.py
"""

DF_CLEANED    = "data/df_cleaned.csv"
HUC2_SHP      = "data/shp_files/Watershed_Boundary_Dataset_HUC_2.gpkg"
LAKES_SHP     = "data/greatLakesShp/greatLakes.shp"

FIG1_MAP        = "figs/fig1_sampling_map_aridity.png"
FIG1_BOXPLOTS   = "figs/fig2b_aridity_boxplots.png"
FIG_S1          = "figs/figS1_climate_gradient.png"
FIG_S1_GEOLOGY  = "figs/figS1_geology_landuse.png"


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

#
# PUBLICATION STYLE
#
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset':   'dejavuserif',
    'font.size':          16,
    'axes.labelsize':     18,
    'axes.titlesize':     19,
    'xtick.labelsize':    15,
    'ytick.labelsize':    15,
    'legend.fontsize':    15,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'axes.linewidth':     1.0,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

GRID_COL  = '#cccccc'
TEXT_COL  = '#222222'
MUTED_COL = '#555555'


#
# FIGURE 1 — LANDSCAPE AND CLIMATE DRIVERS OF GROUNDWATER CO₂
#
def plot_climate_gradient(df, target_col='CO2_aq', save_path=None):
    """
    Twelve-panel figure showing groundwater CO₂ vs landscape/climate drivers.
    Continuous variables: hexbin + running median.
    Zero-inflated land cover variables: boxplots over fixed bins.
    Each panel shows Spearman ρ and n.
    """
    df = df.dropna(subset=[target_col]).copy()
    df = df[df[target_col] > 0].copy()
    df['log_co2'] = np.log10(df[target_col])
    if 'tmp_dc_s' in df.columns:
        df['tmp_C'] = df['tmp_dc_s'] / 10.0
    if 'soc_th_sav' in df.columns:
        df['soc_th_sav'] = df['soc_th_sav'].clip(upper=200)
    if 'Dd' in df.columns:
        df['Dd'] = df['Dd'].clip(upper=10)
    if 'dtb' in df.columns:
        df['dtb'] = df['dtb'].clip(upper=20000)

    # Variables to render as binned boxplots instead of hexbins
    BOX_BINS = {
        'urb_pc_sse': [-0.01, 0, 5, 25, 50, 100],
        'crp_pc_sse': [-0.01, 0, 10, 33, 66, 100],
    }
    BOX_LABELS = {
        'urb_pc_sse': ['0', '0–5', '5–25', '25–50', '50–100'],
        'crp_pc_sse': ['0', '0–10', '10–33', '33–66', '66–100'],
    }

    fig, axes = plt.subplots(4, 3, figsize=(20, 17))
    fig.patch.set_facecolor('white')
    panels = [
        ('ari_ix_sav',  'Aridity Index (PET / P × 100)',        '(a)'),
        ('swc_pc_s',    'Soil Water Content (%)',               '(b)'),
        ('ele_mt_sav',  'Mean elevation',                       '(c)'),
        ('Latitude_x',  'Latitude (°N)',                        '(d)'),
        ('Longitude_x', 'Longitude (°E)',                       '(e)'),
        ('cly_pc_sav',  'Clay Percentage (%)',                  '(f)'),
        ('soc_th_sav',  'Soil Organic Carbon (t ha$^{-1}$)',    '(g)'),
        ('urb_pc_sse',  'Urban Land Cover (%)',                 '(h)'),
        ('crp_pc_sse',  'Crop Land Cover (%)',                  '(i)'),
        ('slp_dg_sav',  'Average slope (m/m)',                  '(j)'),
        ('Dd',          'Drainage Density',                     '(k)'),
        ('dtb',         'Depth to Bedrock (m)',                 '(l)'),
    ]
    for ax, (xcol, xlabel, label) in zip(axes.flat, panels):
        sub = df.dropna(subset=[xcol]).copy()
        if len(sub) == 0:
            ax.text(0.5, 0.5, f'{xcol} not available',
                    transform=ax.transAxes, ha='center')
            continue

        # Spearman computed on raw (un-binned) values for honest correlation
        rho, pval = stats.spearmanr(sub[xcol], sub['log_co2'])

        if xcol in BOX_BINS:
            # Binned boxplot for zero-inflated variables
            edges = BOX_BINS[xcol]
            cats = pd.cut(sub[xcol], bins=edges, labels=BOX_LABELS[xcol],
                          include_lowest=True)
            grouped = [sub.loc[cats == lab, 'log_co2'].values
                       for lab in BOX_LABELS[xcol]]
            counts = [len(g) for g in grouped]
            positions = np.arange(len(BOX_LABELS[xcol]))
            bp = ax.boxplot(grouped, positions=positions, widths=0.6,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=2))
            for patch in bp['boxes']:
                patch.set_facecolor('#fdb863')
                patch.set_edgecolor('#666666')
            # Sample size labels under each box
            ymin, ymax = ax.get_ylim()
            for pos, n in zip(positions, counts):
                ax.text(pos, ymax - 0.01 * (ymax - ymin),
                        f'n={n:,}', ha='center', va='top',
                        fontsize=10, color='#444444')
            ax.set_xticks(positions)
            ax.set_xticklabels(BOX_LABELS[xcol])
            # No colorbar for boxplot panels — pad the layout to match
        else:
            # Hexbin for continuous variables
            hb = ax.hexbin(sub[xcol], sub['log_co2'], gridsize=50,
                           cmap='YlOrRd', mincnt=3,
                           linewidths=0.15, edgecolors='#dddddd')
            bins_q = pd.qcut(sub[xcol], q=20, duplicates='drop')
            med_y = sub.groupby(bins_q)['log_co2'].median()
            med_x = sub.groupby(bins_q)[xcol].median()
            ax.plot(med_x, med_y, 'w-', linewidth=4.5, zorder=4)
            ax.plot(med_x, med_y, 'k-', linewidth=2.5, zorder=5,
                    label='Running median')
            cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
            cb.set_label('Count', fontsize=16)
            cb.ax.tick_params(labelsize=14)

        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(r'log$_{10}$ CO$_2$ (aq) [mol L$^{-1}$]', fontsize=20)
        ax.set_ylim(-6, -2)
        ax.set_title(label, fontweight='bold',fontsize=24, loc='center')
        ax.text(0.97, 0.05,
                f'ρ = {rho:.3f}\n'
                f'n = {len(sub):,}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=20,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.92, edgecolor='#999999', linewidth=1.2))
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)

    fig.suptitle(r'Landscape and Climate Drivers of Groundwater CO$_2$ (aq)',
                 fontsize=26, y=1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved  {save_path}')
    plt.show()
    return fig, axes


#
# HYGEO2 CATEGORY DEFINITIONS
#

HYGEO_CAT = {
    'Major basin':    [11, 12, 13, 14],
    'Complex':        [22, 23, 24, 25],
    'Local/shallow':  [33, 34, 35, 36],
}

AQUIFER_CAT_COLORS = {
    'Major basin':    '#dd6e42',
    'Complex':        '#71816d',
    'Local/shallow':  '#4f6d7a',
}

# Fixed West  East order
HUC2_ORDER = ['18', '17', '16', '15', '14', '13', '10', '09',
              '11', '12', '08', '07', '04', '05', '06', '03', '02', '01']


#
# HELPER: spread overlapping boxplot positions
#
def _spread_positions(raw_pos, min_gap=2.5):
    """Push sorted positions apart so adjacent HUC2 boxes don't overlap."""
    pos = np.array(raw_pos, dtype=float)
    for _ in range(200):
        moved = False
        for i in range(len(pos) - 1):
            gap = pos[i + 1] - pos[i]
            if gap < min_gap:
                shift = (min_gap - gap) / 2.0
                pos[i]     -= shift
                pos[i + 1] += shift
                moved = True
        if not moved:
            break
    return pos.tolist()


#
# FIGURE 2 — MAP + HUC2 CO₂ MARGINAL (coloured by category or aridity)
#
def plot_sampling_map(df, target_col='CO2_aq', huc2_shp=None,
                      lakes_shp=None, color_by='hygeo', save_path=None):
    """
    Standalone map figure with marginal HUC2 CO₂ distributions.

    Top panel:    Scatter of well locations coloured by sample-count bin
                  (1, 2–10, 10–20, 20+) with HUC2 basin outlines,
                  Great Lakes, and HUC2 centroid labels.
    Bottom panel: Boxplots of CO₂(aq) per HUC2 region in fixed
                  West  East order, coloured by dominant category
                  or continuous aridity.

    Parameters
    ----------
    color_by : str, default 'hygeo'
        - 'hygeo'   : dominant HYGEO2 aquifer type (categorical legend)
        - 'aridity' : continuous colorbar from median P/PET
                       (blue = wet, red = dry)

    HUC2 regions 19 and 21 are excluded.
    """
    import geopandas as gpd
    from collections import OrderedDict
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Patch

    #  Data prep
    df = df.dropna(subset=['Latitude_x', 'Longitude_x', target_col]).copy()
    df = df[
        (df['Latitude_x'] >= 24) & (df['Latitude_x'] <= 50) &
        (df['Longitude_x'] >= -125) & (df['Longitude_x'] <= -65)
    ].copy()
    df['HUC2'] = df['HUC2'].astype(str).str.zfill(2)
    df = df[~df['HUC2'].isin(['19', '21'])].copy()

    #  Unique wells + bins
    well_df = df.groupby('MonitoringLocation').agg(
        lat=('Latitude_x', 'first'),
        lon=('Longitude_x', 'first'),
        n_samples=(target_col, 'count'),
    ).reset_index()

    conditions = [
        well_df['n_samples'] == 1,
        (well_df['n_samples'] >= 2)  & (well_df['n_samples'] <= 10),
        (well_df['n_samples'] > 10)  & (well_df['n_samples'] <= 20),
        well_df['n_samples'] > 20,
    ]
    bin_labels = ['1', '2–10', '10–20', '20+']
    well_df['count_bin'] = np.select(conditions, bin_labels, default='1')

    bin_style = OrderedDict([
        ('1',     dict(color='#DAD2BC', s=5,  alpha=0.35, zorder=2)),
        ('2–10',  dict(color='#ADB5BD', s=10, alpha=0.60, zorder=3)),
        ('10–20', dict(color='#495057', s=18, alpha=0.75, zorder=4)),
        ('20+',   dict(color='#212529', s=30, alpha=0.90, zorder=5)),
    ])

    #  HUC2 boxplot data — FIXED ORDER
    huc2_medlon = df.groupby('HUC2')['Longitude_x'].median()
    huc2_order  = [h for h in HUC2_ORDER if h in huc2_medlon.index]

    groups   = [df.loc[df['HUC2'] == h, target_col].values
                for h in huc2_order]
    raw_pos  = [huc2_medlon[h] for h in huc2_order]
    positions = _spread_positions(raw_pos, min_gap=2.5)

    #  Category / aridity assignment per HUC2
    use_continuous_aridity = False
    has_categories = False
    huc2_dominant_cat = {}
    huc2_aridity = {}         # mean ari_ix_sav per HUC2 (continuous mode)
    cat_colors = {}
    cat_order = []
    legend_title = ''

    if color_by == 'aridity':
        ari_col = 'ari_ix_sav'
        if ari_col in df.columns:
            use_continuous_aridity = True
            print("\n  HUC2 mean aridity index (ari_ix_sav = PET/P × 100):")
            print(f"  {'HUC2':>5s}  {'mean ari_ix_sav':>16s}")
            print(f"  {''*5}  {''*16}")
            for huc in huc2_order:
                huc_data = df.loc[df['HUC2'] == huc, ari_col].dropna()
                if len(huc_data) == 0:
                    huc2_aridity[huc] = np.nan
                    continue
                huc2_aridity[huc] = huc_data.mean()
                print(f"  {huc:>5s}  {huc2_aridity[huc]:>16.1f}")
        else:
            print(f" '{ari_col}' not in dataframe — "
                  f"using uniform grey for boxplots")

    else:  # color_by == 'hygeo' (default)
        hygeo_col = 'HYGEO2'
        if hygeo_col in df.columns:
            df['_hygeo_int'] = pd.to_numeric(df[hygeo_col], errors='coerce')
            for huc in huc2_order:
                huc_data = df.loc[df['HUC2'] == huc, '_hygeo_int'].dropna()
                if len(huc_data) == 0:
                    huc2_dominant_cat[huc] = 'Complex'
                    continue
                cat_counts = {}
                for cat_name, codes in HYGEO_CAT.items():
                    cat_counts[cat_name] = huc_data.isin(codes).sum()
                huc2_dominant_cat[huc] = max(cat_counts, key=cat_counts.get)

            has_categories = True
            cat_colors = AQUIFER_CAT_COLORS
            cat_order = ['Major basin', 'Complex', 'Local/shallow']
            legend_title = 'Dominant Aquifer Type'
            for cat in cat_order:
                hucs = [h for h, c in huc2_dominant_cat.items() if c == cat]
                if hucs:
                    print(f"  {cat}: {', '.join(hucs)}")
        else:
            print(f" 'HYGEO2' not in dataframe — "
                  f"using uniform grey for boxplots")

    #  Figure layout
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.2)
    ax_map = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1], sharex=ax_map)
    fig.patch.set_facecolor('white')

    for ax in (ax_map, ax_box):
        ax.set_facecolor('white')
        ax.tick_params(colors=TEXT_COL, labelsize=12)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_edgecolor('#333333')

    #  MAP: HUC2 basin outlines + centroid labels
    if huc2_shp is not None:
        try:
            huc2_gdf = gpd.read_file(huc2_shp)
            huc_col = None
            for cand in ['huc2', 'HUC2', 'HUC_2', 'HUCID',
                         'HUC2_CODE', 'huc_2']:
                if cand in huc2_gdf.columns:
                    huc_col = cand
                    break
            if huc_col is not None:
                huc2_gdf['_huc2_code'] = (
                    huc2_gdf[huc_col].astype(str).str.zfill(2))
                huc2_gdf = huc2_gdf[
                    huc2_gdf['_huc2_code'].astype(int).between(1, 18)]
            if (huc2_gdf.crs is not None
                    and not huc2_gdf.crs.is_geographic):
                huc2_gdf = huc2_gdf.to_crs(epsg=4326)

            huc2_gdf.boundary.plot(
                ax=ax_map, linewidth=1.0,
                edgecolor='#333333', alpha=0.8, zorder=6)

            for _, row in huc2_gdf.iterrows():
                centroid = row.geometry.representative_point()
                code = row.get('_huc2_code', '')
                if (centroid.x < -126 or centroid.x > -64
                        or centroid.y < 24 or centroid.y > 53):
                    continue
                ax_map.text(
                    centroid.x, centroid.y, code,
                    fontsize=9, fontweight='bold', color='black',
                    ha='center', va='center', zorder=10,
                    bbox=dict(boxstyle='round,pad=0.15',
                              facecolor='white', alpha=0.6,
                              edgecolor='none'))

            print(f" {len(huc2_gdf)} HUC2 outlines + labels plotted")
        except Exception as e:
            print(f" HUC2 shapefile error: {e}")
    else:
        print(" huc2_shp not provided — outlines omitted")

    #  MAP: Great Lakes
    if lakes_shp is not None:
        try:
            lakes_gdf = gpd.read_file(lakes_shp)
            if (lakes_gdf.crs is not None
                    and not lakes_gdf.crs.is_geographic):
                lakes_gdf = lakes_gdf.to_crs(epsg=4326)
            lakes_gdf.plot(
                ax=ax_map, facecolor='#B0D4F1', edgecolor='#4A90D9',
                linewidth=0.6, alpha=0.7, zorder=1)
            print(f" Great Lakes plotted ({len(lakes_gdf)} features)")
        except Exception as e:
            print(f" Great Lakes shapefile error: {e}")
    else:
        print(" lakes_shp not provided — lakes omitted")

    #  MAP: scatter points
    for label, style in bin_style.items():
        sub = well_df[well_df['count_bin'] == label]
        ax_map.scatter(
            sub['lon'], sub['lat'],
            s=style['s'], c=style['color'],
            alpha=style['alpha'], edgecolors='none',
            zorder=style['zorder'],
            label=f"{label}  (n = {len(sub):,})")

    ax_map.set_xlim(-126, -64)
    ax_map.set_ylim(24, 53)
    ax_map.set_ylabel('Latitude', fontsize=16)
    ax_map.set_xlabel('Longitude', fontsize=16)
    #ax_map.set_title(
    #    r'Well Locations and HUC2 CO$_2$(aq) Distributions',
    #    fontsize=15, fontweight='bold', loc='left')
    ax_map.set_aspect('equal')
    ax_map.grid(True, color=GRID_COL, linewidth=0.4, alpha=0.4)
    #ax_map.tick_params(labelbottom=False)

    leg = ax_map.legend(
        loc='lower left', fontsize=11, frameon=True,
        fancybox=True, edgecolor='#999999',
        title='Samples per well', title_fontsize=13,
        markerscale=2.0, scatterpoints=1,
        labelspacing=0.8, handletextpad=0.6)
    leg.get_title().set_fontweight('bold')

    ax_map.text(
        0.98, 0.04,
        f'Total wells = {len(well_df):,}\nTotal samples = {len(df):,}',
        transform=ax_map.transAxes, fontsize=12,
        va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='white',
                  alpha=0.92, edgecolor='#999999'))

    #  BOXPLOT
    bp = ax_box.boxplot(
        groups, positions=positions, widths=2.0,
        patch_artist=True,
        medianprops =dict(color='#222222', linewidth=1.8),
        whiskerprops=dict(color='#666666', linewidth=0.9),
        capprops    =dict(color='#666666', linewidth=0.9),
        flierprops  =dict(marker='.', markersize=1.5, alpha=0.25,
                          markerfacecolor='#888888',
                          markeredgecolor='none'),
        showfliers=False, notch=False, manage_ticks=False)

    #  Colour boxes
    if use_continuous_aridity:
        # ari_ix_sav: high = dry, low = wet
        # RdBu: red at low end, blue at high end
        ari_vals = np.array([huc2_aridity.get(h, np.nan)
                             for h in huc2_order])
        valid = ari_vals[~np.isnan(ari_vals)]
        vmin, vmax = valid.min(), valid.max()

        aridity_cmap = cm.get_cmap('RdBu')
        aridity_norm = Normalize(vmin=vmin, vmax=vmax)

        for patch, h in zip(bp['boxes'], huc2_order):
            val = huc2_aridity.get(h, np.nan)
            if np.isnan(val):
                patch.set_facecolor('#cccccc')
            else:
                patch.set_facecolor(aridity_cmap(aridity_norm(val)))
            patch.set_alpha(0.85)
            patch.set_edgecolor('#444444')
            patch.set_linewidth(0.7)

        # Continuous colorbar
        sm = cm.ScalarMappable(cmap=aridity_cmap, norm=aridity_norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_box, pad=0.06, shrink=0.85,
                          aspect=25)
        cb.set_label('Aridity Index (PET/P × 100)',
                     fontsize=12)
        cb.ax.tick_params(labelsize=9)

    elif has_categories:
        for patch, h in zip(bp['boxes'], huc2_order):
            dom_cat = huc2_dominant_cat.get(h, cat_order[0])
            patch.set_facecolor(cat_colors.get(dom_cat, '#cccccc'))
            patch.set_alpha(0.85)
            patch.set_edgecolor('#444444')
            patch.set_linewidth(0.7)

        legend_patches = [
            Patch(facecolor=cat_colors[cat], edgecolor='#444444',
                  linewidth=0.7, alpha=0.85, label=cat)
            for cat in cat_order
            if any(c == cat for c in huc2_dominant_cat.values())
        ]
        leg_box = ax_box.legend(
            handles=legend_patches, loc='lower right',
            fontsize=12, frameon=True, fancybox=True,
            edgecolor='#999999', title=legend_title,
            title_fontsize=10)
        leg_box.get_title().set_fontweight('bold')

    else:
        for patch in bp['boxes']:
            patch.set_facecolor('#cccccc')
            patch.set_alpha(0.85)
            patch.set_edgecolor('#444444')
            patch.set_linewidth(0.7)

    ax_box.set_yscale('log')
    ax_box.set_xlabel('Longitude', fontsize=14)
    ax_box.set_ylabel(
        r'CO$_2$(aq)' + '\n' + r'[mol L$^{-1}$]', fontsize=14)
    ax_box.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)

    #  Secondary y-axis: CO₂ in ppm
    ppm_col = 'CO2_ppm'
    if ppm_col in df.columns and target_col in df.columns:
        # Empirical conversion factor from the data
        both = df[[target_col, ppm_col]].dropna()
        both = both[(both[target_col] > 0) & (both[ppm_col] > 0)]
        if len(both) > 0:
            ratio = (both[ppm_col] / both[target_col]).median()

            ax_ppm = ax_box.twinx()
            ax_ppm.set_yscale('log')
            # Sync limits: convert mol/L limits  ppm
            ylo, yhi = ax_box.get_ylim()
            ax_ppm.set_ylim(ylo * ratio, yhi * ratio)
            ax_ppm.set_ylabel('CO$_2$ (ppm)', fontsize=14)
            ax_ppm.tick_params(labelsize=12)
            ax_ppm.spines['right'].set_visible(True)
            ax_ppm.spines['right'].set_edgecolor('#333333')

    #  HUC2 labels above boxes
    trans = mtransforms.blended_transform_factory(
        ax_box.transData, ax_box.transAxes)

    for h, pos, grp in zip(huc2_order, positions, groups):
        n_h = len(grp)
        ax_box.text(
            pos, 1.08, h,
            transform=trans, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black')
        ax_box.text(
            pos, 1.02, f'n={n_h:,}',
            transform=trans, ha='center', va='bottom',
            fontsize=9, color=MUTED_COL)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.005, hspace=0.2)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        print(f'Saved  {save_path}')
    plt.show()
    return fig

#
# FIGURE 2b — STANDALONE ARIDITY BOXPLOTS
#
def plot_aridity_boxplots(df, target_col='CO2_aq', save_path=None):
    """
    Standalone figure: HUC2 CO₂(aq) boxplots coloured by continuous aridity.
    Same data/logic as the bottom panel of plot_sampling_map(color_by='aridity').
    """
    import matplotlib.transforms as mtransforms

    df = df.dropna(subset=['Latitude_x', 'Longitude_x', target_col]).copy()
    df = df[
        (df['Latitude_x'] >= 24) & (df['Latitude_x'] <= 53) &
        (df['Longitude_x'] >= -125) & (df['Longitude_x'] <= -65)
    ].copy()
    df['HUC2'] = df['HUC2'].astype(str).str.zfill(2)
    df = df[~df['HUC2'].isin(['19', '21'])].copy()

    #  HUC2 boxplot data — FIXED ORDER
    huc2_medlon = df.groupby('HUC2')['Longitude_x'].median()
    huc2_order  = [h for h in HUC2_ORDER if h in huc2_medlon.index]

    groups   = [df.loc[df['HUC2'] == h, target_col].values
                for h in huc2_order]
    raw_pos  = [huc2_medlon[h] for h in huc2_order]
    positions = _spread_positions(raw_pos, min_gap=2.5)

    #  Aridity per HUC2
    ari_col = 'ari_ix_sav'
    huc2_aridity = {}
    for huc in huc2_order:
        huc_data = df.loc[df['HUC2'] == huc, ari_col].dropna()
        huc2_aridity[huc] = huc_data.mean() if len(huc_data) > 0 else np.nan

    ari_vals = np.array([huc2_aridity.get(h, np.nan) for h in huc2_order])
    valid = ari_vals[~np.isnan(ari_vals)]
    vmin, vmax = valid.min(), valid.max()
    aridity_cmap = cm.get_cmap('RdBu')
    aridity_norm = Normalize(vmin=vmin, vmax=vmax)

    #  Figure
    fig, ax_box = plt.subplots(figsize=(24, 6))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(right=0.95)  #  reserve space on right for colorbar
    ax_box.set_facecolor('white')
    ax_box.tick_params(colors=TEXT_COL)
    for sp in ('left', 'bottom'):
        ax_box.spines[sp].set_edgecolor('#333333')

    bp = ax_box.boxplot(
        groups, positions=positions, widths=2.0,
        patch_artist=True,
        medianprops =dict(color='#222222', linewidth=1.8),
        whiskerprops=dict(color='#666666', linewidth=0.9),
        capprops    =dict(color='#666666', linewidth=0.9),
        flierprops  =dict(marker='.', markersize=1.5, alpha=0.25,
                          markerfacecolor='#888888',
                          markeredgecolor='none'),
        showfliers=False, notch=False, manage_ticks=False)

    for patch, h in zip(bp['boxes'], huc2_order):
        val = huc2_aridity.get(h, np.nan)
        if np.isnan(val):
            patch.set_facecolor('#cccccc')
        else:
            patch.set_facecolor(aridity_cmap(aridity_norm(val)))
        patch.set_alpha(0.85)
        patch.set_edgecolor('#444444')
        patch.set_linewidth(0.7)

    # Continuous colorbar
    sm = cm.ScalarMappable(cmap=aridity_cmap, norm=aridity_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_box, pad=0.06, shrink=0.85, aspect=25)  # pad 0.02  0.06
    cb.set_label('Aridity Index (PET/P × 100)', fontsize=24)
    cb.ax.tick_params(labelsize=20)

    ax_box.set_yscale('log')
    ax_box.set_xlabel('Longitude', fontsize=26)
    ax_box.set_ylabel(
        r'CO$_2$(aq)' + '\n' + r'(mol L$^{-1}$)', fontsize=26)
    ax_box.tick_params(labelsize=20)
    ax_box.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)

    #  Secondary y-axis: CO₂ in ppm
    ppm_col = 'CO2_ppm'
    if ppm_col in df.columns and target_col in df.columns:
        both = df[[target_col, ppm_col]].dropna()
        both = both[(both[target_col] > 0) & (both[ppm_col] > 0)]
        if len(both) > 0:
            ratio = (both[ppm_col] / both[target_col]).median()
            ax_ppm = ax_box.twinx()
            ax_ppm.set_yscale('log')
            ylo, yhi = ax_box.get_ylim()
            ax_ppm.set_ylim(ylo * ratio, yhi * ratio)
            ax_ppm.set_ylabel('CO$_2$ (ppm)', fontsize=26)
            ax_ppm.tick_params(labelsize=20)
            ax_ppm.spines['right'].set_visible(True)
            ax_ppm.spines['right'].set_edgecolor('#333333')

    #  HUC2 labels above boxes
    trans = mtransforms.blended_transform_factory(
        ax_box.transData, ax_box.transAxes)

    for h, pos, grp in zip(huc2_order, positions, groups):
        n_h = len(grp)
        ax_box.text(
            pos, 1.08, h,
            transform=trans, ha='center', va='bottom',
            fontsize=18, fontweight='bold', color='black')
        ax_box.text(
            pos, 1.02, f'n={n_h:,}',
            transform=trans, ha='center', va='bottom',
            fontsize=12, color=MUTED_COL)

    # ax_box.set_title(
    #     r'HUC2 CO$_2$(aq) Distributions Coloured by Aridity',
    #     fontsize=14, fontweight='bold', loc='left')

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved  {save_path}')
    plt.show()
    return fig

#
# FIGURE 3 — GEOLOGY & LAND USE EMERGE IN UPPER QUANTILES
#
def plot_geology_landuse_by_quantile(df, target_col='CO2_aq',
                                     n_quantiles=8, save_path=None):
    """
    Four-panel figure:
      (a) Normalised feature profiles across CO₂ quantile bins
      (b) Karst fraction vs CO₂  (hexbin)
      (c) Cropland fraction vs CO₂
      (d) Soil organic carbon vs CO₂
    """
    df = df.dropna(subset=[target_col]).copy()
    df = df[df[target_col] > 0].copy()
    df['log_co2'] = np.log10(df[target_col])
    df['q_bin'] = pd.qcut(df[target_col], q=n_quantiles,
                           labels=False, duplicates='drop') + 1

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('white')
    for ax in axes.flat:
        ax.set_facecolor('white')
        ax.tick_params(colors=TEXT_COL)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_edgecolor('#333333')

    #  (a) feature profiles across quantile bins
    ax = axes[0, 0]
    features = {
        'kar_pc_sse': ('Karst %',             '#E74C3C', 'o'),
        'crp_pc_sse': ('Cropland %',          '#27AE60', 's'),
        'soc_th_sav': ('SOC (t ha$^{-1}$)',   '#8E44AD', '^'),
        'wet_pc_s01': ('Wetland %',           '#2E86C1', 'D'),
        'lka_pc_sse': ('Lake %',              '#F39C12', 'v'),
    }
    for feat, (label, color, marker) in features.items():
        if feat not in df.columns:
            continue
        means = df.groupby('q_bin')[feat].mean()
        vals  = means.values.astype(float)
        vrange = vals.max() - vals.min()
        normed = (vals - vals.min()) / vrange if vrange > 0 else vals * 0
        ax.plot(means.index, normed, marker=marker, color=color,
                linewidth=2.5, markersize=7, label=label, zorder=3)

    ax.set_xlabel(r'CO$_2$ Quantile Bin', fontsize=13)
    ax.set_ylabel('Normalised Mean (0–1)', fontsize=13)
    ax.set_title('(a)  Landscape Features by CO$_2$ Quantile',
                 fontsize=14, fontweight='bold', loc='left')
    ax.legend(fontsize=10, frameon=True, fancybox=True,
              edgecolor='#999999', loc='upper left')
    ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)
    ax.set_xticks(range(1, n_quantiles + 1))
    ax.set_xticklabels([f'Q{i}' for i in range(1, n_quantiles + 1)])

    #  (b)–(d) hexbin panels
    hex_panels = [
        ('kar_pc_sse', 'Karst Fraction (%)',              '(b)', 'YlOrRd'),
        ('crp_pc_sse', 'Cropland Fraction (%)',           '(c)', 'YlGn'),
        ('soc_th_sav', 'Soil Organic Carbon (t ha$^{-1}$)', '(d)', 'PuBuGn'),
    ]
    for ax, (xcol, xlabel, label, cname) in zip(axes.flat[1:], hex_panels):
        sub = df.dropna(subset=[xcol]).copy()
        if len(sub) == 0:
            continue

        hb = ax.hexbin(sub[xcol], sub['log_co2'], gridsize=45,
                       cmap=cname, mincnt=3,
                       linewidths=0.15, edgecolors='#dddddd')

        rho, pval = stats.spearmanr(sub[xcol], sub['log_co2'])

        bq   = pd.qcut(sub[xcol], q=20, duplicates='drop')
        medy = sub.groupby(bq)['log_co2'].median()
        medx = sub.groupby(bq)[xcol].median()
        ax.plot(medx, medy, 'w-', linewidth=4.5, zorder=4)
        ax.plot(medx, medy, 'k-', linewidth=2.5, zorder=5)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(r'log$_{10}$ CO$_2$ (aq) [mol L$^{-1}$]', fontsize=13)
        ax.set_title(label, fontsize=20, fontweight='bold', loc='center')
        ax.text(0.97, 0.05,
                f'ρ = {rho:.3f}\nn = {len(sub):,}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.92, edgecolor='#999999', linewidth=1.2))
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.5)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label('Count', fontsize=10)
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        r'Geology and Land-Use Controls on Groundwater CO$_2$ (aq)',
        fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved  {save_path}')
    plt.show()
    return fig, axes


#
if __name__ == '__main__':
    df_cleaned = pd.read_csv(DF_CLEANED)
    # df_cleaned = pd.read_parquet(...)

    plot_climate_gradient(df_cleaned,
                          save_path=FIG_S1)

    # Figure 2 — aquifer coloring (default)
    # plot_sampling_map(
    #     df_cleaned,
    #     huc2_shp=HUC2_SHP,
    #     lakes_shp=LAKES_SHP,
    #     color_by='hygeo',
    #     save_path=FIG1_MAP_HYGEO,
    # )

    # Figure 2 variant — continuous aridity coloring
    plot_sampling_map(
        df_cleaned,
        huc2_shp=HUC2_SHP,
        lakes_shp=LAKES_SHP,
        color_by='aridity',
        save_path=FIG1_MAP,
    )
# Standalone aridity boxplots
    plot_aridity_boxplots(
        df_cleaned,
        save_path=FIG1_BOXPLOTS,
    )
    plot_geology_landuse_by_quantile(df_cleaned,
        save_path=FIG_S1_GEOLOGY)
