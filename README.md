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
│   ├── 07_out_of_sample_validation.ipynb  # OOS validation on calm period
│   ├── 08_signal_backtesting.ipynb       # Signal backtesting and validation
│   └── 09_llm_market_interpreter.ipynb   # LLM regime assessment and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data.py                        # Data loading, cleaning, alignment
│   ├── entropy.py                     # Entropy and information measures
│   ├── microstructure.py              # Trade flow analysis utilities
│   ├── phase_transitions.py           # Regime detection framework
│   ├── metastability.py               # Free-energy landscape analysis
│   ├── visualisation.py               # Plotting with consistent styling
│   ├── signals.py                     # Trading signal generation (5 signals)
│   ├── backtesting.py                 # Event-driven signal backtester
│   └── llm_interpreter.py            # LLM market state interpreter (dual backend)
│
├── tests/                             # Pytest test suite
│   ├── conftest.py                    # Shared fixtures (synthetic data)
│   ├── test_data.py
│   ├── test_entropy.py
│   ├── test_microstructure.py
│   ├── test_phase_transitions.py
│   ├── test_metastability.py
│   ├── test_visualisation.py
│   ├── test_signals.py
│   ├── test_backtesting.py
│   └── test_llm_interpreter.py
│
├── prompts/
│   └── market_interpreter_v1.md      # LLM system prompt (versioned)
│
├── data/                              # .gitignored — raw + processed data
└── outputs/
    ├── figures/                       # Exported key figures
    └── llm_reports/                   # LLM-generated market briefings
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

# Launch Jupyter and run notebooks in order (01 → 09)
jupyter lab
```

**Data note:** Raw trade data is not included in the repository (too large). Run Notebook 01 to download and process the data from Binance's public data repository.

**LLM backends for Notebook 09:**
- **Anthropic API (primary):** Set the `ANTHROPIC_API_KEY` environment variable. Best accuracy.
- **Local Ollama (fallback):** Install [Ollama](https://ollama.ai), then `ollama pull llama3.1`. Free, no data leaves your machine.
- Notebook 09 works with either backend and gracefully degrades if neither is available.

## Testing

```bash
# Run all unit tests (no LLM needed)
pytest tests/ -v --tb=short

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run LLM integration tests (requires active backend)
pytest tests/ -v -m integration --backend ollama

# Run a specific module's tests
pytest tests/test_entropy.py -v
```

176 tests across 9 modules (171 unit + 5 integration), all unit tests using synthetic data and completing in under 30 seconds. Integration tests require a running LLM backend.

## Out-of-Sample Validation

Notebook 07 downloads a calm/ranging week (Feb 15–21, 2026) and replicates the full analysis pipeline to test generalisability. Key questions addressed:

- Do entropy distributions and TE leadership hold outside the crash period?
- Does the low-entropy signal retain predictive accuracy in calm markets?
- Do metastable levels on Binance predict Bybit support/resistance with a time lag?

See [07_out_of_sample_validation.ipynb](notebooks/07_out_of_sample_validation.ipynb) for the side-by-side comparison table and honest generalisability assessment.

## Signal Backtesting

Notebook 08 tests whether the 5 entropy-based signals actually generate tradeable alpha using a lightweight event-driven backtester:

- **Individual signal performance:** Each signal backtested with random-entry baseline comparison, reporting Sharpe ratio, win rate, max drawdown, and profit factor
- **Combined strategy:** Priority-based signal hierarchy (crash type > ACF risk > low entropy > TE flip > metastable)
- **Walk-forward validation:** 2-day train / 1-day test rolling windows with threshold re-calibration to prevent overfitting; IS→OOS critical test
- **Sensitivity analysis:** Parameter sweeps for entropy percentile, transaction costs (breakeven identification), and position sizing
- **Honest assessment:** IS vs OOS degradation ratios, overfitting detection, practical barriers to live implementation

See [08_signal_backtesting.ipynb](notebooks/08_signal_backtesting.ipynb) for the full analysis and honest narrative on what works and what doesn't.

## LLM Market State Interpreter

Notebook 09 integrates an LLM (Anthropic Claude or local Ollama) to translate physics-derived features into natural language regime assessments with structured trading recommendations:

- **Dual backend architecture:** Anthropic API (primary, best accuracy) and local Ollama (free fallback)
- **System prompt engineering:** Encodes the complete physics-to-finance mapping, regime definitions, trading signals, and few-shot examples in a versioned prompt (`prompts/market_interpreter_v1.md`)
- **Evaluation:** Regime classification accuracy, prompting strategy comparison (zero-shot vs few-shot vs CoT vs ablation), backend comparison, consistency analysis
- **Sample market briefings:** LLM-generated reports for key market moments saved to `outputs/llm_reports/`

The LLM does not replace quantitative signals — it translates them into a format that portfolio managers and risk committees can act on without understanding transfer entropy or free-energy landscapes.

See [09_llm_market_interpreter.ipynb](notebooks/09_llm_market_interpreter.ipynb) for the full analysis and [outputs/llm_reports/](outputs/llm_reports/) for sample briefings.

## Project Evolution

This project began as a 7-day analysis of cross-venue information flow during a BTC crash period. It has since been extended through five phases into a portfolio-grade research project:

- **Phase 0 — Cleanup:** Removed assessment-specific artefacts, migrated figures to `outputs/`, added MIT licence, restructured repo for professional presentation.
- **Phase 1 — Test suite:** Added 176 tests (171 unit + 5 integration) across 9 test modules with synthetic data fixtures. All unit tests complete in under 30 seconds.
- **Phase 2 — Out-of-sample validation (Notebook 07):** Downloaded a calm/ranging week (Feb 15–21) and replicated the full pipeline. Key finding: entropy distributions and $\tau_{\mathrm{int}}$ generalise across regimes; TE leadership does not.
- **Phase 3 — Signal backtesting (Notebook 08):** Built an event-driven backtester with walk-forward validation. Honest result: most signals overfit (70%+ IS→OOS degradation); only the crash type classifier retains positive OOS Sharpe.
- **Phase 4 — LLM interpreter (Notebook 09):** Integrated an LLM (Anthropic Claude / local Ollama) to translate physics features into natural language regime assessments. 75% rolling accuracy; chain-of-thought prompting reaches 88%; physics framework ablation confirms the mapping adds genuine value.
- **Phase 5 — Final polish:** Extended `report.md` with sections 7–11 covering all new results, updated limitations, and consolidated trading implications.

## Author

**Wadoud Charbak**, MSci Physics, Imperial College London
