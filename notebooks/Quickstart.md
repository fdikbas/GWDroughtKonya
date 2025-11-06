# Quickstart

This short guide shows how to run the pipeline and where outputs land.

```bash
conda env create -f environment.yml
conda activate konyagw
python scripts/konya_gw_all_in_one_EN.py
```

Outputs:
- `out_figs_EN/` – main figures (TIFF, 400 dpi)
- `out_tables_EN/` – tables (CSV): trend values, GW‑NDSPI, drought events, GEV return levels
- `out_SGI_spatial_EN/` – spatial maps (IDW)
- `out_SGI_plots_EN/` – per‑well SGI series

Swap in your own data by replacing the files in `data/`:
- **Excel** `well_data_imputed.xlsx` with columns `Yıl`, `Ay`, then one column per well ID.
- **Coordinates** `Konya.Well.Coordinates.csv` with columns `Well,X,Y,GroundElev(optional)` in EPSG:4326.
