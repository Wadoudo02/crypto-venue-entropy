# Entropy Project Upgrade: Full Implementation Plan

**Author:** Wadoud Charbak
**Date:** March 2026
**Repo:** github.com/Wadoudo02/crypto-venue-entropy

---

## Overview

This plan upgrades the crypto-venue-entropy project from a one-week assessment piece into a portfolio-grade research project suitable for quant research/trading roles. The work is structured in five sequential phases, each building on the last.

**New notebooks:** 3 (numbered 07, 08, 09)
**New src modules:** 4 (`validation.py`, `backtesting.py`, `llm_interpreter.py`, `signals.py`)
**New directories:** `tests/`, `outputs/llm_reports/`
**Updated files:** `README.md`, `report.md`, `requirements.txt`, `.gitignore`
**Removed files:** `Note_to_Assessor.md`, `.vscode/`

---

## Phase 0: De-Assessment Cleanup

**Goal:** Remove all traces of the Fasanara assessment context. The project should read as independent research, not a submission.

### 0.1 File removals
- [x] Delete `Note_to_Assessor.md`
- [x] Delete `.vscode/` directory
- [x] Verify neither is referenced anywhere in notebooks or report

### 0.2 Content audit
- [x] Search all notebooks (01 through 06) for any mention of: "Fasanara", "assessment", "assessor", "Mike Perkins", "exercise", "submission", "deliverable"
- [x] Replace any found references with neutral language (e.g., "this research project" instead of "this assessment") — none found; notebooks were already clean
- [x] Check `report.md` for the same terms and clean up — no assessment-specific terms found
- [x] Check `README.md` for any assessment-specific language — already clean

### 0.3 Figures migration
- [x] Move `figures/` into `outputs/figures/` so all generated artefacts live under one roof
- [x] Update all notebook references from `figures/` to `outputs/figures/`
- [x] Update `report.md` image paths accordingly
- [x] Update `.gitignore` if needed (figures should remain tracked, data should not)

### 0.4 Repo hygiene
- [x] Add `LICENSE` file (MIT licence)
- [x] Add `.vscode/` to `.gitignore`
- [x] Ensure `data/` and any `.parquet` files are properly gitignored
- [x] Ensure no API keys, personal paths, or machine-specific paths are hardcoded anywhere
- [x] Remove any `Wadoud_Finance_CV.pdf` if present in the repo — not present

### 0.5 README update (first pass)
- [x] Remove any assessment framing — already clean
- [x] Add a "What's New" or "Project Evolution" section placeholder (to be filled after all phases)
- [x] Ensure setup instructions still work (conda env, pip install, notebook order)

**Completed:** commit `d0ae067`.

---

## Phase 1: Test Suite for All `src/` Modules

**Goal:** Add a `tests/` directory with pytest-based unit tests covering every module in `src/`. This is both good practice and a signal to interviewers that you treat code quality seriously.

### 1.1 Directory structure
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures (sample data, mock datasets)
├── test_data.py             # Tests for src/data.py
├── test_entropy.py          # Tests for src/entropy.py
├── test_microstructure.py   # Tests for src/microstructure.py
├── test_phase_transitions.py # Tests for src/phase_transitions.py
├── test_metastability.py    # Tests for src/metastability.py
├── test_visualisation.py    # Tests for src/visualisation.py
├── test_signals.py          # Tests for src/signals.py (new)
├── test_backtesting.py      # Tests for src/backtesting.py (new)
└── test_llm_interpreter.py  # Tests for src/llm_interpreter.py (new)
```

### 1.2 Test philosophy
- Tests use small synthetic datasets, NOT the full 69M+ trade dataset
- Each test should run in under 2 seconds
- Tests verify correctness of computation, not visual output
- `conftest.py` provides shared fixtures: a small DataFrame of ~1000 synthetic trades with known properties, so expected outputs can be computed by hand

### 1.3 Tests for existing modules

**`test_data.py`** (src/data.py)
- Test column standardisation: input raw Binance/Bybit format, verify output schema matches expected columns
- Test tick rule classification: provide known trade sequences, verify buy/sell signs are correct
- Test log return computation: known prices in, verify log returns match manual calculation
- Test inter-trade duration computation: known timestamps, verify durations
- Test handling of edge cases: empty DataFrame, single row, duplicate timestamps

**`test_entropy.py`** (src/entropy.py)
- Test Shannon entropy on known distributions:
  - All buys (entropy = 0)
  - Perfectly balanced 50/50 (entropy = 1.0 for binary)
  - Known unbalanced ratio (e.g., 75/25, verify against manual H calculation)
- Test transfer entropy:
  - Independent random sequences (TE should be near zero)
  - Perfectly coupled sequences with known lag (TE should be high in one direction)
  - Symmetric sequences (TE should be roughly equal both ways)
- Test mutual information:
  - Identical sequences (MI should be maximal)
  - Independent sequences (MI should be near zero)
- Test volume-weighted entropy: verify weighting is applied correctly vs unweighted

**`test_microstructure.py`** (src/microstructure.py)
- Test autocorrelation function on known AR(1) process: verify decay matches theoretical value
- Test integrated autocorrelation time: known series with known tau_int
- Test cross-correlation: perfectly correlated vs independent series
- Test trade sign persistence: alternating signs (low persistence) vs constant signs (high persistence)

**`test_phase_transitions.py`** (src/phase_transitions.py)
- Test order parameter (net imbalance) computation: known trade sequences
- Test susceptibility (variance of imbalance): constant imbalance (zero variance) vs oscillating
- Test entropy discontinuity detection: inject a synthetic entropy series with a known sharp drop, verify detection
- Test regime classification: provide feature vectors for known regimes, verify labels

**`test_metastability.py`** (src/metastability.py)
- Test free-energy landscape computation: provide a bimodal price distribution, verify two wells are found
- Test well depth calculation: known distribution with analytically computable well depth
- Test metastable level detection: known price series that consolidates at specific levels
- Test overlap with traditional S/R: provide known S/R levels and metastable levels, verify overlap calculation

**`test_visualisation.py`** (src/visualisation.py)
- Lighter tests: verify functions run without error on sample data
- Verify returned figure objects are valid matplotlib Figure/Axes
- Do NOT test visual appearance (that is fragile and unnecessary)

**`test_signals.py`** (src/signals.py)
- Test low_entropy_signal: inject entropy series with known percentiles, verify signal fires at correct timestamps
- Test te_leadership_flip: inject net TE series with a known sign flip lasting 2+ windows, verify detection
- Test te_leadership_flip: verify no signal when flip lasts only 1 window
- Test acf_risk_signal: inject ACF series where values exceed 2x trailing median at known points, verify signal
- Test crash_type_classifier: provide feature vectors matching known crash signatures (info-driven vs mechanical), verify correct classification
- Test metastable_order_signal: provide well depths above and below threshold, verify correct filtering and timeout logic
- Test edge cases: empty series, all-NaN series, single-element series

**`test_backtesting.py`** (src/backtesting.py)
- Test with a single known trade: fixed entry/exit prices, verify P&L calculation matches manual computation
- Test transaction cost application: verify costs are deducted correctly (2 bps round-trip)
- Test Sharpe ratio calculation: provide a known equity curve with analytically computable Sharpe
- Test max drawdown calculation: provide equity curve with a known peak-to-trough, verify drawdown matches
- Test win rate: provide a set of trades with known outcomes, verify win rate calculation
- Test no-trade scenario: no signals fired, verify backtester returns zero trades and flat equity curve
- Test position sizing: verify fractional position sizing is applied correctly

**`test_llm_interpreter.py`** (src/llm_interpreter.py)
- **Backend abstraction tests** (no LLM needed):
  - Test MarketSnapshot serialisation to JSON and back
  - Test RegimeAssessment parsing from valid JSON string
  - Test RegimeAssessment parsing fails gracefully on malformed JSON
  - Test that MarketInterpreter accepts both AnthropicBackend and OllamaBackend
- **Integration tests** (require a running Ollama instance OR are skipped):
  - Mark with `@pytest.mark.integration` so they can be excluded in CI
  - Use local Ollama as the default backend (no API key needed)
  - Test single snapshot assessment: send a known crisis snapshot, verify output parses to valid RegimeAssessment
  - Test that confidence is lowered when contradictory features are provided
  - Test that output JSON schema is always valid (even if regime classification is wrong)
  - Test batch assessment: send 3 snapshots, verify 3 valid assessments returned
- **Pytest configuration**: `conftest.py` provides a `--backend` flag:
  - `pytest tests/ -v` runs all unit tests (no LLM needed)
  - `pytest tests/ -v -m integration --backend ollama` runs integration tests against local Ollama
  - `pytest tests/ -v -m integration --backend anthropic` runs integration tests against Anthropic API

### 1.4 How to run
```bash
# From repo root
pytest tests/ -v --tb=short
```

Add to README: a "Testing" section with the above command.

**Estimated time:** 4-6 hours. Tedious but high-value for credibility.

**Completed:** commits `9947db6` through `93fe0c1`. 90 tests across 6 test files, all passing in ~4 seconds on synthetic data. README updated with Testing section.

---

## Phase 2: Extended Data and Out-of-Sample Validation (Notebook 07)

**Completed:** commits `a04781f` through `(latest)`. Added `src/signals.py` (5 signal functions), `tests/test_signals.py` (29 tests, all passing), and `notebooks/07_out_of_sample_validation.ipynb` (5 sections: data acquisition, metric replication, comparison table, generalisability narrative, cross-venue metastability). README updated. 119 total tests passing.

**Goal:** Download at least one additional week of data from a different market regime and test whether the key findings from the original analysis generalise.

### 2.1 New module: `src/signals.py`
Before the notebook, extract the 5 trading signals into a reusable module:

```python
# src/signals.py
"""
Signal generation from entropy and microstructure features.
Each function takes a feature DataFrame and returns a signal Series.
"""

def low_entropy_signal(entropy_series, threshold_pctile=5):
    """Signal 1: Shannon entropy below Nth percentile."""
    ...

def te_leadership_flip(net_te_series, consecutive_windows=2):
    """Signal 2: Transfer entropy leadership reversal."""
    ...

def acf_risk_signal(acf_time_series, multiplier=2.0):
    """Signal 3: ACF time exceeding Nx trailing median."""
    ...

def crash_type_classifier(entropy, te_binance, te_bybit, acf_time):
    """Signal 4: Information-driven vs mechanical crash classification."""
    ...

def metastable_order_signal(well_depths, threshold=5.0, stale_timeout_s=41):
    """Signal 5: Passive limit order placement at deep metastable levels."""
    ...
```

### 2.2 Notebook 07: Extended Data and Out-of-Sample Validation

**Filename:** `notebooks/07_out_of_sample_validation.ipynb`

**Structure:**

#### Section 1: Data Acquisition for New Period
- Download 1 week of Binance + Bybit BTC-USDT perp trades from a different regime
- Target: a ranging/calm period (not another crash) to test generalisability
- Candidate periods: check recent BTC history for a low-vol week
- Use the exact same `src/data.py` pipeline (no changes to processing)
- Basic sanity checks: trade counts, price range, venue split

#### Section 2: Replicate Key Metrics
Run the existing analysis pipeline on the new data and report the same headline numbers:
- Binance vs Bybit Shannon entropy means
- Transfer entropy leadership split (% windows Binance leads at k=1)
- Mutual information decay rate
- Integrated autocorrelation time distribution
- Low-entropy signal accuracy (5th percentile, 5-min forward return)
- Number of metastable levels, well depth distribution

#### Section 3: Side-by-Side Comparison Table
A clear table comparing original period vs new period for every key metric. This is the centrepiece of the notebook.

| Metric | Original (Jan 30 - Feb 6) | New Period (dates) | Change |
|--------|---------------------------|---------------------|--------|
| Binance mean entropy | 0.989 | ? | |
| Binance leadership (k=1) | 59.4% | ? | |
| MI 1s decay | 86% | ? | |
| Low-entropy signal accuracy | 88.1% | ? | |
| ACF-vol correlation (rho) | 0.34 | ? | |
| Vol multiplier (>90th pctile) | 1.65x | ? | |
| Metastable levels found | 98 | ? | |

#### Section 4: What Generalises and What Doesn't
Honest narrative assessment. Possible outcomes:
- **Best case:** Key signals generalise, strengthening all claims
- **Mixed case:** Some signals generalise (e.g., TE leadership holds but low-entropy signal is weaker). This is fine and interesting
- **Worst case:** Nothing generalises, meaning original findings were regime-specific. Still valuable as an honest result

#### Section 5: Cross-Venue Metastability (Preliminary)
- Compute metastable levels separately for Binance and Bybit
- Test whether Binance levels predict Bybit support/resistance with a time lag
- This connects Phase 3 (information leadership) with Phase 5 (free-energy landscapes)
- Report correlation and lag structure, even if weak

**Trading implication for each section** as per the golden rule.

**Expected outputs:**
- Side-by-side comparison table (the key deliverable)
- Figures comparing distributions across periods
- Honest narrative on generalisability
- Cross-venue metastability preliminary results

**Estimated time:** 6-10 hours (depends heavily on data download and processing time).

---

## Phase 3: Signal Backtesting (Notebook 08)

**Goal:** Answer the question every interviewer will ask: "Do these signals actually make money?"

### 3.1 New module: `src/backtesting.py`

```python
# src/backtesting.py
"""
Lightweight event-driven backtester for entropy-based signals.
Not a full production framework, but sufficient for signal validation.
"""

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # 'long' or 'short' or 'reduce'
    entry_price: float
    exit_price: float
    pnl: float
    signal_type: str

class SignalBacktester:
    def __init__(self, prices, signals, transaction_cost_bps=2.0):
        """
        prices: pd.Series of mid-prices at signal resolution
        signals: pd.DataFrame with columns [timestamp, signal_type, direction, ...]
        transaction_cost_bps: round-trip cost in basis points
        """
        ...

    def run(self) -> BacktestResult:
        """Execute backtest, return results."""
        ...

@dataclass
class BacktestResult:
    trades: List[Trade]
    equity_curve: pd.Series
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    num_trades: int
    
    def summary(self) -> str:
        """Return formatted summary string."""
        ...
```

### 3.2 Notebook 08: Signal Backtesting

**Filename:** `notebooks/08_signal_backtesting.ipynb`

**Structure:**

#### Section 1: Backtesting Framework
- Explain the methodology: event-driven, no look-ahead bias, transaction costs included
- Define entry/exit rules for each signal (derived from the 5 trading signals)
- Define position sizing: fixed fractional (1% risk per signal)
- Define transaction costs: 2 bps round-trip (realistic for perps on Binance/Bybit)

#### Section 2: Individual Signal Performance
For each of the 5 signals, run the backtest and report:
- Number of trades triggered
- Win rate
- Average P&L per trade
- Sharpe ratio
- Maximum drawdown
- Equity curve plot
- Comparison against a naive baseline (e.g., random entry with same holding period)

Present as a summary table:

| Signal | Trades | Win Rate | Sharpe | Max DD | Profit Factor |
|--------|--------|----------|--------|--------|---------------|
| Low entropy | ? | ? | ? | ? | ? |
| TE leadership flip | ? | ? | ? | ? | ? |
| ACF risk | ? | ? | ? | ? | ? |
| Crash type | ? | ? | ? | ? | ? |
| Metastable orders | ? | ? | ? | ? | ? |

#### Section 3: Combined Strategy
- Test a combined strategy using multiple signals
- Define priority/weighting when signals conflict
- Report combined equity curve and metrics

#### Section 4: Walk-Forward Validation
- Split the full dataset (original + extended) into rolling train/test windows
- Train signal thresholds on train window, test on next window
- Report out-of-sample metrics across all test windows
- This is critical: it prevents overfitting of signal thresholds

#### Section 5: Sensitivity Analysis
- How sensitive are results to threshold choices?
- How sensitive to transaction cost assumptions?
- How sensitive to position sizing?

#### Section 6: Honest Assessment
This is arguably the most important section of the notebook. A realistic but mediocre Sharpe ratio is worth far more in an interview than a suspiciously good one. Overfitting and look-ahead bias are the first things a quant interviewer will probe for.

- Which signals work and which don't? If a signal fails the backtest, **do not hide it**. Highlight it, explain why it failed, and discuss what that tells you about the signal's nature. A signal that works in-sample but fails out-of-sample is a genuine finding about overfitting.
- What are the practical barriers to live implementation? (latency, data availability, execution slippage)
- How do results compare to the in-sample metrics from the original analysis? Quantify the degradation.
- If the combined Sharpe is below 1.0 or even negative, report it honestly. The value of this notebook is the methodology and the intellectual honesty, not the P&L number.

**Trading implication for each section.**

**Expected outputs:**
- Individual signal performance table
- Combined strategy equity curve
- Walk-forward out-of-sample results
- Sensitivity analysis charts
- Honest narrative on what works

**Estimated time:** 8-12 hours. This is the most technically demanding notebook.

---

## Phase 4: LLM Market State Interpreter (Notebook 09)

**Goal:** Demonstrate LLM integration for interpretable market intelligence. The LLM ingests structured physics-derived features and produces natural language regime assessments with structured trading recommendations.

### 4.1 New module: `src/llm_interpreter.py`

The module supports **two backends**: the Anthropic API (Claude) and a local Ollama instance. This is not just convenience; it demonstrates proper software engineering (abstracting over providers) and means anyone cloning the repo can run Notebook 09 without an API key by using a local model.

```python
# src/llm_interpreter.py
"""
LLM-based market state interpreter with dual backend support.
Supports: Anthropic API (Claude) and local Ollama models.
Ingests entropy/microstructure features, outputs structured regime assessments.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class MarketSnapshot:
    """A single point-in-time market state from the physics pipeline."""
    timestamp: str
    binance_entropy: float
    bybit_entropy: float
    binance_entropy_percentile: float
    net_transfer_entropy: float
    te_leader: str  # 'binance' or 'bybit'
    te_leader_consecutive_windows: int
    acf_time: float
    acf_time_percentile: float
    acf_trailing_median: float
    realised_vol_30m: float
    order_imbalance: float
    susceptibility: float
    nearest_metastable_level: float
    nearest_well_depth: float
    price: float
    return_5m: float

@dataclass
class RegimeAssessment:
    """Structured output from the LLM interpreter."""
    timestamp: str
    regime: str  # 'calm', 'transitional', 'crisis_information', 'crisis_mechanical'
    confidence: float  # 0 to 1
    confidence_rationale: str  # Why this confidence level (especially if low)
    crash_type: Optional[str]  # 'information_driven', 'mechanical', None
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    conflicting_signals: list[str]  # Any features pointing in opposite directions
    recommended_actions: list[str]
    reasoning: str  # Chain-of-thought explanation
    physics_summary: str  # Plain English summary of feature state
    key_features_driving_assessment: list[str]


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt and return the raw text response."""
        ...


class AnthropicBackend(LLMBackend):
    """Anthropic API backend (Claude)."""
    
    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def complete(self, system_prompt: str, user_message: str) -> str:
        ...


class OllamaBackend(LLMBackend):
    """Local Ollama backend. Default model: llama3.1 or mistral."""
    
    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def complete(self, system_prompt: str, user_message: str) -> str:
        ...


class MarketInterpreter:
    def __init__(self, backend: LLMBackend):
        self.backend = backend
    
    def assess(self, snapshot: MarketSnapshot) -> RegimeAssessment:
        """Send a market snapshot to the LLM and parse structured response."""
        ...
    
    def assess_batch(self, snapshots: list[MarketSnapshot]) -> list[RegimeAssessment]:
        """Assess multiple snapshots (e.g., rolling window across dataset)."""
        ...
    
    def evaluate_accuracy(
        self, 
        assessments: list[RegimeAssessment], 
        ground_truth_regimes: list[str]
    ) -> dict:
        """Compare LLM regime calls against ground truth labels."""
        ...


# Usage:
# backend = AnthropicBackend(api_key="...")   # Cloud
# backend = OllamaBackend(model="llama3.1")  # Local
# interpreter = MarketInterpreter(backend)
# result = interpreter.assess(snapshot)
```

### 4.2 System prompt design

The system prompt is the intellectual core of this component. It must encode:

1. **The physics-to-finance mapping table** (all 10 rows)
2. **Threshold definitions** for each regime:
   - Calm: entropy > 0.95, ACF time < median, no TE dominance
   - Transitional: mixed signals, moderate ACF elevation
   - Crisis (information-driven): entropy collapse (< 5th pctile), clear TE leader, ACF spike
   - Crisis (mechanical): normal entropy, bidirectional TE, liquidation signatures
3. **The 5 trading signals** with exact threshold values
4. **Few-shot examples**: 3-4 annotated snapshots from the original dataset with known regime labels and correct assessments
5. **Output schema** enforced via the prompt (JSON format matching RegimeAssessment)
6. **Contradictory signal handling** (critical for real-world credibility):
   - The LLM must explicitly identify when features point in opposing directions (e.g., low entropy suggesting crisis but normal ACF time suggesting calm)
   - When signals conflict, the LLM must lower its confidence score and explain why in `confidence_rationale`
   - The `conflicting_signals` field must list the specific contradictions
   - Example instruction: "If entropy suggests crisis but ACF time is below median, report confidence below 0.5 and list both signals in conflicting_signals. Do not force a high-confidence classification when the physics is ambiguous."
   - This matters because ~56% of windows in the original data were classified as Transitional precisely because of mixed signals. The LLM must handle ambiguity honestly, not paper over it.

The prompt engineering itself is a deliverable. Store it as `prompts/market_interpreter_v1.md` for version control and iteration.

### 4.3 Notebook 09: LLM Market State Interpreter

**Filename:** `notebooks/09_llm_market_interpreter.ipynb`

**Structure:**

#### Section 1: System Prompt Design
- Display and explain the full system prompt
- Justify design choices (why this framework encoding, why these few-shot examples)
- Discuss the structured output schema

#### Section 2: Single Snapshot Demonstration
- Pick 3-4 interesting moments from the dataset:
  - A calm period
  - The Jan 31 information-driven crash
  - The Feb 5-6 mechanical crash
  - A transitional period
- Show the full LLM input and output for each
- Annotate: does the LLM correctly identify the regime and recommend appropriate actions?

#### Section 3: Rolling Window Evaluation
- Run the interpreter across the full dataset in rolling 30-minute windows
- Generate regime assessments for every window
- Compute ground truth labels from the actual subsequent price action
- Report accuracy metrics:
  - Overall regime classification accuracy
  - Crash type identification accuracy
  - Latency: how many windows after a regime shift does the LLM correctly identify it?
  - Confusion matrix: which regimes get misclassified as what?

#### Section 4: Prompting Strategy Comparison
- Compare performance across prompting strategies:
  - Zero-shot (no examples, just the framework)
  - Few-shot (3-4 examples)
  - Chain-of-thought (explicit reasoning steps)
  - Ablation: remove the physics framework, just give raw numbers. Does physics context help?
- Report accuracy for each strategy
- This is genuinely interesting and publishable-quality analysis

#### Section 5: Backend Comparison (Anthropic vs Ollama)
- Run the same set of snapshots through both backends
- Compare: accuracy, consistency, latency, output quality
- This is a practical question: can a local model do this job, or is a cloud API required?
- Report as a comparison table. Even if the local model is worse, that's a useful finding.

#### Section 6: LLM Consistency and Failure Modes
- Run the same snapshot through the LLM 5 times. How consistent are the outputs?
- Identify failure modes: when does the LLM get it wrong? Are failures systematic?
- Test edge cases: what happens with extreme values? Missing features?
- Specifically test: does the LLM correctly lower confidence when given contradictory features?

#### Section 7: Cost and Latency Analysis
- Tokens per assessment (input + output)
- Cost per assessment at current API pricing (Anthropic) vs free (Ollama)
- Latency per assessment for each backend
- Feasibility for real-time use

#### Section 8: Sample Market Briefings
- Generate 3-5 polished natural language market state reports for key moments
- These are the "portfolio pieces" that demonstrate the system's output quality
- Save as individual markdown files in `outputs/llm_reports/`

**Trading implication for each section.**

**Expected outputs:**
- Regime classification accuracy metrics
- Prompting strategy comparison table (the ablation study)
- Backend comparison table (Anthropic vs Ollama)
- Consistency analysis
- Sample LLM-generated market briefings (saved to `outputs/llm_reports/`)
- Cost/latency analysis per backend

### 4.4 Sample LLM output (what a generated report looks like)

```json
{
  "timestamp": "2026-01-31T14:30:00Z",
  "regime": "crisis_information",
  "confidence": 0.92,
  "confidence_rationale": "All four primary indicators (entropy, TE leadership, ACF time, well depth) align consistently toward an information-driven crisis. No contradictory signals detected.",
  "crash_type": "information_driven",
  "risk_level": "extreme",
  "conflicting_signals": [],
  "recommended_actions": [
    "Reduce passive exposure on both venues immediately",
    "Shift execution to Binance (current information leader)",
    "Do NOT provide liquidity at metastable levels (well depths degrading)",
    "Widen execution bands by 2x baseline"
  ],
  "reasoning": "Shannon entropy on Binance has collapsed to the 2nd percentile (H=0.61), indicating highly directional order flow. Transfer entropy shows clear Binance leadership sustained for 4 consecutive windows. The integrated autocorrelation time has spiked to 3.2x its trailing median, signalling critical slowing down. This signature matches the information-driven crash pattern: a coordinated sell-off led by informed flow on the dominant venue, not a mechanical liquidation cascade. The nearest metastable level at $78,200 has a well depth of only 1.3, suggesting it will not hold.",
  "physics_summary": "The system has undergone a phase transition from a disordered (high-entropy) state to an ordered (low-entropy) state with strong directional coupling between venues. This is analogous to spontaneous magnetisation below the critical temperature.",
  "key_features_driving_assessment": [
    "binance_entropy_percentile: 2nd (extreme low)",
    "te_leader: binance (4 consecutive windows)",
    "acf_time: 3.2x trailing median (critical slowing down)",
    "nearest_well_depth: 1.3 (degrading support)"
  ]
}
```

**Example of a low-confidence output with conflicting signals:**

```json
{
  "timestamp": "2026-02-02T08:15:00Z",
  "regime": "transitional",
  "confidence": 0.35,
  "confidence_rationale": "Entropy suggests emerging directional pressure (18th percentile) but ACF time remains below median, inconsistent with a regime shift. Order imbalance is elevated but susceptibility is low. The physics is ambiguous.",
  "crash_type": null,
  "risk_level": "medium",
  "conflicting_signals": [
    "Entropy at 18th percentile suggests directional pressure, but ACF time is below median (no critical slowing down)",
    "Order imbalance elevated at -0.3, but susceptibility is low (market not sensitive to shocks)"
  ],
  "recommended_actions": [
    "Maintain current position sizing (no clear signal to reduce)",
    "Monitor entropy and ACF time for convergence before acting",
    "Keep passive orders at metastable levels with well depth > 5.0"
  ],
  "reasoning": "...",
  "physics_summary": "...",
  "key_features_driving_assessment": ["..."]
}
```

**Estimated time:** 8-12 hours.

---

## Phase 5: Final Outputs and Polish

### 5.1 Updated `report.md`

The existing report covers Phases 1-6 (original work). Extend it with new sections:

**New sections to add:**

7. **Out-of-Sample Validation** (from Notebook 07)
   - Summary of extended dataset
   - Side-by-side comparison table
   - Generalisability assessment
   - Cross-venue metastability results

8. **Signal Backtesting** (from Notebook 08)
   - Backtesting methodology
   - Individual signal performance table
   - Combined strategy results
   - Walk-forward validation results
   - Honest assessment of practical viability

9. **LLM Market State Interpreter** (from Notebook 09)
   - System design and prompt engineering approach
   - Evaluation results (accuracy, latency, cost)
   - Prompting strategy comparison
   - Sample outputs
   - Discussion: where LLMs add value vs where they don't

10. **Updated Limitations and Future Work**
    - Revised limitations incorporating new findings
    - Remaining gaps
    - Production deployment considerations

11. **Updated Trading Implications**
    - Consolidated table of all signals with out-of-sample validation status
    - Which signals survived backtesting
    - How the LLM interpreter complements quantitative signals

### 5.2 LLM-generated reports directory

```
outputs/
└── llm_reports/
    ├── README.md                              # Explains what these are
    ├── calm_period_2026-02-03T10:00.md        # Sample calm regime report
    ├── crash_info_2026-01-31T14:30.md         # Jan 31 crash report
    ├── crash_mech_2026-02-05T22:00.md         # Feb 5-6 crash report
    └── transitional_2026-02-02T08:00.md       # Transitional regime report
```

Each file is a formatted markdown version of the LLM's JSON output, rendered as a readable market briefing. These serve as portfolio pieces: someone browsing the repo can click into `outputs/llm_reports/` and immediately see what the system produces.

These are generated by Notebook 09, not manually written. The notebook includes code that converts the JSON RegimeAssessment into a clean markdown report and saves it.

### 5.3 Updated README.md

Final README structure:

```markdown
# crypto-venue-entropy

Statistical Mechanics of Cross-Venue Information Flow in Bitcoin Perpetual Futures

## Motivation
[existing, cleaned of assessment language]

## Key Findings
[existing, plus new OOS and backtesting headlines]

## Methodology
[existing physics mapping table]

## Project Structure
[updated to include new notebooks, modules, tests, outputs]

## New: Out-of-Sample Validation
[brief summary with link to Notebook 07]

## New: Signal Backtesting
[brief summary with headline Sharpe/win rate, link to Notebook 08]

## New: LLM Market State Interpreter
[brief summary, link to Notebook 09, link to sample reports]

## Setup and Reproduction
[existing, plus note about LLM backends for Notebook 09:
- Anthropic API: set ANTHROPIC_API_KEY environment variable
- Local Ollama: install Ollama, pull a model (e.g., ollama pull llama3.1)
- Notebook 09 works with either backend; Ollama is the zero-cost default]

## Testing
pytest tests/ -v                                    # Unit tests (no LLM needed)
pytest tests/ -v -m integration --backend ollama    # Integration tests with local Ollama

## Author
Wadoud Charbak, MSci Physics, Imperial College London
```

### 5.4 Updated requirements.txt

Add new dependencies:
```
# Existing
numpy
pandas
matplotlib
scipy
...

# New (Phase 4)
anthropic>=0.40.0    # Cloud LLM backend
ollama>=0.4.0        # Local LLM backend
```

### 5.5 Final repo structure

```
crypto-venue-entropy/
├── README.md
├── LICENSE                                    # NEW
├── report.md                                  # UPDATED
├── requirements.txt                           # UPDATED
├── .gitignore                                 # UPDATED
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_entropy_analysis.ipynb
│   ├── 04_phase_transitions.ipynb
│   ├── 05_metastability.ipynb
│   ├── 06_synthesis.ipynb
│   ├── 07_out_of_sample_validation.ipynb      # NEW
│   ├── 08_signal_backtesting.ipynb            # NEW
│   └── 09_llm_market_interpreter.ipynb        # NEW
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── entropy.py
│   ├── microstructure.py
│   ├── phase_transitions.py
│   ├── metastability.py
│   ├── visualisation.py
│   ├── signals.py                             # NEW
│   ├── backtesting.py                         # NEW
│   └── llm_interpreter.py                     # NEW
│
├── prompts/
│   └── market_interpreter_v1.md               # NEW
│
├── tests/                                     # NEW
│   ├── __init__.py
│   ├── conftest.py
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
├── outputs/
│   ├── figures/                               # MOVED from root
│   └── llm_reports/                           # NEW
│       ├── README.md
│       └── [generated market briefings]
│
├── data/                                      # .gitignored
```

---

## Execution Order and Dependencies

```
Phase 0 (cleanup)
    │
    v
Phase 1 (tests for existing src/)
    │
    v
Phase 2 (Notebook 07: OOS validation)  ← requires: new data downloaded
    │                                      produces: validated metrics, new figures
    v
Phase 3 (Notebook 08: backtesting)     ← requires: src/signals.py, both datasets
    │                                      produces: P&L results, equity curves
    v
Phase 4 (Notebook 09: LLM interpreter) ← requires: Anthropic API key, all features computed
    │                                      produces: regime assessments, sample reports
    v
Phase 5 (final polish)                 ← requires: all previous phases complete
                                          produces: updated report, README, outputs/
```

Phases 0 and 1 can be done in parallel.
Phases 2 and 3 are sequential (backtesting needs OOS data).
Phase 4 can technically start in parallel with Phase 3 but is cleaner after.
Phase 5 must be last.

---

## What Success Looks Like

When complete, the project should:

1. **Pass the 30-second GitHub scan:** Clean repo, no assessment artefacts, clear README, visible tests directory, professional structure
2. **Pass the 5-minute deep read:** OOS validation shows intellectual honesty, backtesting shows practical relevance, LLM integration shows modern AI skills
3. **Survive a 30-minute interview grilling:** Every number has OOS context, every signal has a P&L, the LLM component is defensible ("I used it for what LLMs are actually good at"), negative results are owned transparently
4. **Demonstrate distinct skills from the trading bot:** The bot = classical ML + execution. This project = physics-based analysis + LLM interpretation + rigorous validation

---

## Estimated Total Time

| Phase | Estimated Hours |
|-------|----------------|
| Phase 0: Cleanup | 1-2 |
| Phase 1: Tests | 4-6 |
| Phase 2: OOS Validation (NB 07) | 6-10 |
| Phase 3: Backtesting (NB 08) | 8-12 |
| Phase 4: LLM Interpreter (NB 09) | 8-12 |
| Phase 5: Polish | 3-4 |
| **Total** | **30-46 hours** |

At intense pace (your "week that equals a month"), this is achievable in 5-7 days of focused work.

---

## Workflow Reminder

For each phase:
1. We design the detailed prompt/plan together here
2. Save as markdown, pass to Claude Code
3. Claude Code generates the notebook/module/tests
4. You upload the outputs back here for rigorous review
5. We compile fixes into a precise prompt for Claude Code
6. Repeat until clean

The golden rule still applies: **every analysis section ends with "The trading implication is..." followed by a quantitative, actionable statement.**
