# KonyaGW — Groundwater Drought Diagnostics (SGI • MK–Sen • GW-NDSPI • GEV)

_Reproducible pipeline and synthetic demo dataset for groundwater drought analytics on a Konya-like semi‑arid basin._  
This repository ships a **complete, publication‑oriented workflow**: monthly well series → SGI → drought events → MK–Sen trends (+FDR) → change‑points (Pettitt) → spatial IDW maps → persistence (GW‑NDSPI) → extremes (GEV, T=10/25/50).

> **Data provenance.** All inputs in this repo are **synthetic** and safe to share. They mimic realistic depletion patterns but **are not real observations**. Replace them with your own monitoring files to analyze a real basin (see Quickstart).

---

## Contents

```
konya-gw-ndspi-release/
├─ data/
│  ├─ well_data_imputed_SYNTHETIC.xlsx        # Monthly well levels (DTW), 2016–2024 (Turkish columns: Yıl, Ay)
│  ├─ well_data_imputed_SYNTHETIC.csv         # CSV twin of the XLSX
│  └─ Konya.Well.Coordinates_SYNTHETIC.csv    # Well IDs + lon/lat (EPSG:4326)
├─ scripts/
│  └─ konya_gw_all_in_one_EN.py               # End‑to‑end analysis & figure export
├─ notebooks/
│  └─ Quickstart.md                            # Markdown quickstart (copy/paste friendly)
├─ docs/                                       # Space for manuscript figures/notes (optional)
├─ LICENSE, CITATION.cff, CONTRIBUTING.md, requirements.txt, environment.yml, .gitignore
└─ README.md
```

Key outputs (when you run the script) mirror the paper’s structure:
- `out_figs_EN/*.tiff` – publication‑ready figures (400 dpi)
- `out_tables_EN/*.csv` – event catalogs, trend tables, return levels
- `out_SGI_spatial_EN/*.tiff` – spatial IDW maps (trends, GW‑NDSPI, extremes)
- `out_SGI_plots_EN/*.tiff` – per‑well SGI series

Methodological choices and figure directories align with the manuscript code base fileciteturn28file0.

---

## Quickstart

### 1) Create environment
```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate konyagw

# OR pip
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Run the pipeline
```bash
python scripts/konya_gw_all_in_one_EN.py
```
By default the script looks for:
- `data/well_data_imputed_SYNTHETIC.xlsx`
- `data/Konya.Well.Coordinates_SYNTHETIC.csv`

All outputs are written under `out_*` folders at 400 dpi with stable styling (publication‑ready).

### 3) Swap in your own data
- Keep the **same column names** in your Excel: `Yıl`, `Ay`, then one column per well (IDs as headers).  
- Provide coordinates in `Konya.Well.Coordinates_*.csv` with columns: `Well, X, Y, GroundElev(optional)` in **EPSG:4326** (lon/lat).  
- Optional basin boundary (GeoJSON/SHP) can be referenced inside the script if available.

---

## Data dictionary

**`well_data_imputed_SYNTHETIC.(xlsx|csv)`**  
- `Yıl`: year (int), `Ay`: month (1–12)  
- One column per monitoring well (string well IDs). Values are **depth‑to‑water (m bgs)**, negative‑valued and **decreasing with time** to emulate depletion.

**`Konya.Well.Coordinates_SYNTHETIC.csv`**  
- `Well`: ID (string), `X`: longitude (EPSG:4326), `Y`: latitude (EPSG:4326), `GroundElev`: optional meters a.s.l.

---

## Reproducibility & licensing

- Code is released under **GNU GPL v3.0** (see `LICENSE`).  
- Synthetic data are released under **CC BY 4.0** within this repository.  
- Please cite this repository and the associated manuscript if you use the workflow (see `CITATION.cff`).

---

## Troubleshooting

- **Maps look blank / basemap errors** → Ensure `contextily` can fetch tiles; if offline, the script falls back to plain cartography.  
- **Geopandas errors** → Use the provided Conda env; system GEOS/GDAL mismatch is a common issue.  
- **Different coordinate systems** → Provide lon/lat (EPSG:4326). The script handles projection to web mercator internally for mapping.

---

## Acknowledgements

This repository packages a generalized version of the analysis workflow used for semi‑arid basins with groundwater‑drought stress (Konya‑style). The approach combines standardized indices, robust trend tests, persistence diagnostics, and return‑level mapping to support policy‑relevant decisions.

