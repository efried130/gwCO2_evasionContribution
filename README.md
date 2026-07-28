# Groundwater CO2 Contribution to Stream Evasion

Code accompanying *Low Groundwater Carbon Dioxide Contributions to Stream Evasion
in the United States*. These scripts span from water quality portal extraction, 
PHREEQC calculation, model, prediction, evasion contribution analysis, and figures.

## Pipeline order

| Script | Description |
|---|---|
| `dataRetrieval_GW_co2_o2_updated2024.R` | Call data from WQP using dataRetrieval
| `phreeqcInput4.txt` | The specifcations and input for PHREEQC model
| `phreeqc_gwCO2.R` | The PHREEQC model call
| `data_prep.py` | Assemble WQP/PHREEQC + ERA5/GLDAS/HydroATLAS training table, clean outliers
| `model.py` | Train XGBoost model + single-feature ablation | Table S2 |
| `predict.py` | Gridded 0.1-deg monthly CO2(aq) prediction with conformal uncertainty
| `gw_flux.py` | Groundwater CO2 flux per HUC2 (CO2 x runoff x area) with uncertainty
| `contribution_saccardi.py` | Aggregate Saccardi et al. (2024) reach fluxes to HUC2; contribution % |
| `contribution_liu.py` | Liu et al. (2022) ephemeral + ice corrections; contribution % vs all three efflux datasets |
| `figures_main.py` | Figure 1 (sampling map + aridity boxplots) and Figure S1 panels |
| `figures_si.py` | Figure 3 (HUC2 contribution choropleth), Fig S4 barplot, Fig S2b runoff map |


## Paths

Each script has a small configuration block at the top. Inputs are expected under
`data/` and figures are written to `figs/`. Adjust these to your local layout.

## Data

Training data: Zenodo 10.5281/zenodo.21632888. Predictor sources (WQP, HydroATLAS,
ERA5-Land, WHYMAP) and stream efflux estimates are available from their original
publications.
