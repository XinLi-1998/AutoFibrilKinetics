# AutoFibrilKinetics

This repository accompanies our study of glucagon glycoform mixtures and provides the analysis scripts used to extract two kinetic parameters from paired `Trp` and `ThT` fluorescence data:

- `lag time_analysis.py`: calculates lag time by tangent extrapolation
- `elongation_rate_analysis.py`: calculates the apparent elongation rate `k_app (min^-1)`

The repository is intended as a compact, reproducible companion package containing the analysis code and the source data table required to rerun the main kinetic calculations.

Both scripts read the combined raw-data workbook directly:

- `Table S1. Raw fluorescence kinetic data of ThT and Trp under various conditions..xlsx`

## Repository Layout

```text
AutoFibrilKinetics/
├── Table S1. Raw fluorescence kinetic data of ThT and Trp under various conditions..xlsx
├── lag time_analysis.py
├── elongation_rate_analysis.py
├── README.md
└── LICENSE.txt
```

## Requirements

The scripts were written for a standard scientific Python environment. Install the required packages with:

```bash
pip install pandas numpy scipy openpyxl
```

## Quick Start

Place the raw data workbook in the repository root, next to the scripts:

```text
Table S1. Raw fluorescence kinetic data of ThT and Trp under various conditions..xlsx
```

Run lag-time analysis for one dataset:

```bash
python "lag time_analysis.py" --dataset 3.0-3
```

Run elongation-rate analysis for all three datasets:

```bash
python elongation_rate_analysis.py
```

If the workbook is stored elsewhere, specify its location with `--source`:

```bash
python "lag time_analysis.py" --dataset 3.0-3 --source "D:\path\to\Table S1. Raw fluorescence kinetic data of ThT and Trp under various conditions..xlsx"
```

## Command Options

- `lag time_analysis.py`
  - `--dataset 3.0-1|3.0-2|3.0-3`
  - `--source <path_to_table_s1>`
  - `--output-dir <output_folder>`
- `elongation_rate_analysis.py`
  - `--datasets 3.0-1 3.0-2 3.0-3`
  - `--source <path_to_table_s1>`
  - `--output-dir <output_folder>`

## Outputs

- Lag-time analysis:
  - `lag_time_results_3.0-1.xlsx/.csv`
  - `lag_time_results_3.0-2.xlsx/.csv`
  - `lag_time_results_3.0-3.xlsx/.csv`
- Elongation-rate analysis:
  - `elongation_rate_results_3.0-1.xlsx/.csv`
  - `elongation_rate_results_3.0-2.xlsx/.csv`
  - `elongation_rate_results_3.0-3.xlsx/.csv`

The Excel outputs contain replicate-level values, summary statistics, and automatically flagged outliers or review notes, whereas the CSV outputs provide compact export tables for downstream reporting.

## Notes

- `3.0-1`, `3.0-2`, and `3.0-3` correspond to 20%, 33%, and 50% glycoform conditions, respectively.
- The scripts read the required data directly from `Figure 4_G1-G7`, `Figure S9_G8-G14`, and `Figure S10_G15-G21` in Table S1.
- No intermediate input workbooks are created.
- `Trp` is treated as the primary kinetic readout, with `ThT` used only as a limited corrective reference when the paired trace satisfies the built-in confidence criteria.

## Reproducibility Scope

- This repository is designed to reproduce the primary lag-time and elongation-rate calculations for the three main glycoform-mixing conditions reported in the manuscript.
- The scripts operate directly on the source data table included in the repository and do not require any manual reformatting before execution.
- Output locations can be redirected with `--output-dir` if a separate results folder is preferred.

## Troubleshooting

- `FileNotFoundError`: confirm that the Table S1 workbook is located beside the scripts, or pass it explicitly with `--source`.
- Missing package error: reinstall dependencies with `pip install pandas numpy scipy openpyxl`.
- Wrong dataset selection: use `--dataset 3.0-1|3.0-2|3.0-3` for lag time, or `--datasets ...` for elongation rate.
- Custom output location: use `--output-dir`.
