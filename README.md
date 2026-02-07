# crypto-venue-entropy

**Statistical mechanics of cross-venue information flow in crypto perpetual futures markets.**

## Motivation

Cross-venue price discovery in crypto markets is poorly understood at the microstructure level. This project applies entropy measures and phase transition detection from statistical mechanics to quantify how information flows between exchanges and when those flow patterns undergo regime shifts — with explicit trading implications for a cross-venue HFT desk.

## Key Findings

*Results will be populated upon completion of the analysis.*

## Methodology

We treat multi-venue trade flow as a statistical system and map market observables to thermodynamic quantities:

| Market Observable | Physics Analogue |
|---|---|
| Trade sign entropy | Disorder / randomness |
| Transfer entropy (venue A → B) | Directional information flow |
| Realised volatility | Temperature |
| Net order flow imbalance | Order parameter (magnetisation) |
| Variance of imbalance | Susceptibility |
| ACF decay scale | Correlation length |
| Price density landscape | Free-energy landscape |

Key techniques: Shannon entropy of trade signs, transfer entropy for directional causality, correlation length divergence for critical slowing down, and free-energy analogues for metastable price level identification.

## Repository Structure

```
crypto-venue-entropy/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Data download, cleaning, alignment
│   ├── 02_exploratory_analysis.ipynb  # Basic microstructure stats
│   ├── 03_entropy_analysis.ipynb      # Shannon entropy, transfer entropy
│   ├── 04_phase_transitions.ipynb     # Regime detection, critical signatures
│   ├── 05_metastability.ipynb         # Free-energy landscape analysis
│   └── 06_synthesis.ipynb             # Combined findings and conclusions
│
├── src/
│   ├── __init__.py
│   ├── data.py                        # Data loading, cleaning, alignment
│   ├── entropy.py                     # Entropy and information measures
│   ├── microstructure.py              # Trade flow analysis utilities
│   ├── phase_transitions.py           # Regime detection framework
│   ├── metastability.py               # Free-energy landscape analysis
│   └── visualisation.py               # Plotting with consistent styling
│
├── data/                              # .gitignored — raw + processed data
├── figures/                           # Exported key figures
└── report/
    └── report.md                      # Written report
```

## Setup & Reproduction

```bash
# Clone the repository
git clone https://github.com/wadoudcharbak/crypto-venue-entropy.git
cd crypto-venue-entropy

# Create and activate the conda environment
conda create -n crypto-entropy python=3.11
conda activate crypto-entropy

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter and run notebooks in order (01 → 06)
jupyter lab
```

**Data note:** Raw trade data is not included in the repository (too large). Run Notebook 01 to download and process the data from Binance's public data repository.

## Author

**Wadoud Charbak**
