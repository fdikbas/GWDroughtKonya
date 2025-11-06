# Contributing

Thanks for your interest in improving KonyaGW!

## How to contribute
- Open an issue describing a bug or proposal.
- Submit a pull request from a feature branch.
- Keep PRs focused and include before/after notes or screenshots for figure changes.

## Dev setup
```
conda env create -f environment.yml
conda activate konyagw
pre-commit hooks (optional) coming soon
```

## Code style
- Python 3.11+, type hints encouraged.
- Prefer pure-Python + SciPy/NumPy/GeoPandas; keep dependencies minimal.
- Figures: 400 dpi exports, consistent color semantics across years.

## Legal
- Do not contribute or attach any non-public/commercial datasets.
- Synthetic demo data in `data/` are safe to share and CC BY 4.0.
