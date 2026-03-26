# crypto-venue-entropy

**Statistical Mechanics of Cross-Venue Information Flow in Bitcoin Perpetual Futures**

Applying entropy measures and phase transition detection to quantify how information flows between crypto exchanges and when those flow patterns undergo regime shifts.

## Motivation

Cross-venue price discovery in crypto markets is poorly understood at the microstructure level. This project applies statistical mechanics to 69.4M Binance and 35M Bybit BTC-USDT perpetual futures trades (Jan 30 to Feb 6, 2026) during a major crash period ($84K to $62K), with explicit trading implications for a cross-venue HFT desk.

## Key Findings

- **Information leadership:** Transfer entropy reveals Binance leads information flow in 59.4% of 30-minute windows (at $k=1$), resolving a hierarchy invisible to linear cross-correlation. Mutual information drops 86% within 1 second, defining a sub-second exploitation window.
- **Early warning signal:** The integrated autocorrelation time ($\tau_{\mathrm{int}}$) correlates with 30-minute forward volatility at $\rho = 0.34$; when elevated above the 90th percentile, subsequent volatility is 1.65$\times$ baseline.
- **Low-entropy signal:** When Binance Shannon entropy drops below the 5th percentile, 88.1% of signals precede |return| > 0.05% within 5 minutes.
- **Two crash types identified:** Jan 31 crash (entropy collapse, sharp $\tau_{\mathrm{int}}$ spike, clear Binance leadership) is information-driven; Feb 5-6 crash (normal entropy, bidirectional flow, liquidation cascades) is mechanically-driven. Each requires a fundamentally different trading response.
- **Physics validates traditional S/R:** 98 metastable levels from free-energy landscapes overlap 90% with traditional support/resistance, while adding quantitative well depth (0.8 to 7.5) and temporal evolution that traditional methods lack.

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
├── LICENSE                            # MIT licence
├── report.md                          # Written report
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Data download, cleaning, alignment
│   ├── 02_exploratory_analysis.ipynb  # Basic microstructure stats
│   ├── 03_entropy_analysis.ipynb      # Shannon entropy, transfer entropy
│   ├── 04_phase_transitions.ipynb     # Regime detection, critical signatures
│   ├── 05_metastability.ipynb         # Free-energy landscape analysis
│   ├── 06_synthesis.ipynb             # Combined findings and conclusions
│   └── 07_out_of_sample_validation.ipynb  # OOS validation on calm period
│
├── src/
│   ├── __init__.py
│   ├── data.py                        # Data loading, cleaning, alignment
│   ├── entropy.py                     # Entropy and information measures
│   ├── microstructure.py              # Trade flow analysis utilities
│   ├── phase_transitions.py           # Regime detection framework
│   ├── metastability.py               # Free-energy landscape analysis
│   ├── visualisation.py               # Plotting with consistent styling
│   └── signals.py                     # Trading signal generation (5 signals)
│
├── tests/                             # Pytest test suite
│   ├── conftest.py                    # Shared fixtures (synthetic data)
│   ├── test_data.py
│   ├── test_entropy.py
│   ├── test_microstructure.py
│   ├── test_phase_transitions.py
│   ├── test_metastability.py
│   ├── test_visualisation.py
│   └── test_signals.py
│
├── data/                              # .gitignored — raw + processed data
└── outputs/
    └── figures/                       # Exported key figures
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

# Launch Jupyter and run notebooks in order (01 → 07)
jupyter lab
```

**Data note:** Raw trade data is not included in the repository (too large). Run Notebook 01 to download and process the data from Binance's public data repository.

## Testing

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific module's tests
pytest tests/test_entropy.py -v
```

Tests use small synthetic datasets and complete in under 30 seconds.

## Out-of-Sample Validation

Notebook 07 downloads a calm/ranging week (Feb 15–21, 2026) and replicates the full analysis pipeline to test generalisability. Key questions addressed:

- Do entropy distributions and TE leadership hold outside the crash period?
- Does the low-entropy signal retain predictive accuracy in calm markets?
- Do metastable levels on Binance predict Bybit support/resistance with a time lag?

See [07_out_of_sample_validation.ipynb](notebooks/07_out_of_sample_validation.ipynb) for the side-by-side comparison table and honest generalisability assessment.

## Project Evolution

*This section will be updated as the project develops beyond its initial scope.*

## Author

**Wadoud Charbak**, MSci Physics, Imperial College London
