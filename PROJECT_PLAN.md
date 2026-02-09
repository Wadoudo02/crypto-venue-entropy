# Fasanara Digital Technical Exercise — Project Plan
## Statistical Mechanics of Cross-Venue Information Flow in Crypto Markets

**Candidate:** Wadoud Charbak  
**Assessor:** Mike Perkins, Head of HFT, Fasanara Digital  
**Deliverables:** Jupyter Notebook with embedded analysis + short written report + GitHub repo  

---

## 1. Project Overview

### One-Sentence Summary
Apply entropy measures and phase transition detection from statistical mechanics to quantify how information flows between crypto exchanges and when those flow patterns undergo regime shifts — with explicit trading implications throughout.

### Core Questions We Are Answering
1. **Where is informed trading happening?** → Shannon entropy of trade sign sequences at each venue
2. **How does information flow between venues?** → Transfer entropy to measure directional causality
3. **When do market regimes shift?** → Phase transition signatures (entropy discontinuities, correlation length divergence)
4. **What are the quasi-stable price levels?** → Metastability analysis using free-energy analogues
5. **So what?** → Every finding tied back to a concrete trading implication for a cross-venue HFT desk

### The Golden Rule
> Every section ends with: "The trading implication is..."  
> This is how we keep the physics grounded and Mike engaged.

---

## 2. GitHub Setup

### Repository Name
**`crypto-venue-entropy`**

Why this name:
- Short, memorable, descriptive
- Avoids overly academic names like "statistical-mechanics-microstructure-analysis"
- Avoids overly generic names like "fasanara-exercise" or "quant-project"
- Signals the core novelty (entropy applied to venue dynamics) without being obscure
- Professional and clean for LinkedIn sharing

### Repository Structure
```
crypto-venue-entropy/
│
├── README.md                          # Project overview, setup instructions, key findings summary
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Standard Python gitignore
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Data download, cleaning, alignment
│   ├── 02_exploratory_analysis.ipynb  # Basic microstructure stats, initial visualisations
│   ├── 03_entropy_analysis.ipynb      # Shannon entropy, transfer entropy, information flow
│   ├── 04_phase_transitions.ipynb     # Regime detection, correlation lengths, critical signatures
│   ├── 05_metastability.ipynb         # Quasi-stable levels, free-energy landscape
│   └── 06_synthesis.ipynb             # Combined findings, trading implications, conclusions
│
├── src/
│   ├── __init__.py
│   ├── data.py                        # Data loading, cleaning, alignment utilities
│   ├── entropy.py                     # Shannon entropy, transfer entropy, mutual information
│   ├── microstructure.py              # Trade sign classification, lead-lag, Hasbrouck shares
│   ├── phase_transitions.py           # Correlation length, susceptibility, regime detection
│   ├── metastability.py               # Free-energy landscape, quasi-stable level detection
│   └── visualisation.py               # Plotting utilities, consistent styling
│
├── data/                              # .gitignored — raw data stored locally, not in repo
│   ├── raw/
│   └── processed/
│
├── figures/                           # Key figures exported for the report
│
└── report/
    └── report.md                      # Short written report (can also be PDF if we want)
```

### README Structure
The README should be polished and serve as a standalone summary. Structure:
1. **Title + one-line summary**
2. **Motivation** — why this matters for HFT (2-3 sentences)
3. **Key Findings** — 3-4 bullet points with the headline results
4. **Methodology** — brief overview of the physics framework
5. **Repository Structure** — quick guide to navigating the repo
6. **Setup & Reproduction** — how to install deps and run notebooks
7. **Author** — name, links

### Git Practices
- Clean, descriptive commit messages (not "update notebook" but "add transfer entropy computation with 1-min rolling windows")
- Do NOT commit data files (add data/ to .gitignore)
- Commit regularly so the history shows steady, organised progress
- Final commit before submission: make sure all notebooks run cleanly top-to-bottom

---

## 3. Data Strategy

### Target Data
- **Asset:** BTC-USDT perpetual futures (primary), with BTC-USDT spot as secondary reference
- **Venues:** Binance, Bybit, OKX (three venues minimum for meaningful cross-venue analysis)
- **Time period:** 1-2 weeks of recent data (enough to capture both calm and volatile periods)
- **Granularity:** Individual trade-level data (timestamp, price, size, side)

### Data Sources
| Venue   | Source | URL | Notes |
|---------|--------|-----|-------|
| Binance | Public historical data | https://data.binance.vision/?prefix=data/futures/um/daily/trades/BTCUSDT/ | Well-structured CSVs, easy to download |
| Bybit   | Historical data page | https://www.bybit.com/derivatives/en/history-data | May need manual selection |
| OKX     | Direct URL pattern | `https://www.okx.com/cdn/okex/traderecords/trades/monthly/YYYYMM/allfuture-trades-YYYY-MM-DD.zip` | Replace date placeholders |

### Data Processing Pipeline
1. Download raw trade data for each venue
2. Standardise column names and formats across venues
3. Align timestamps (all to UTC, millisecond precision)
4. Classify trade sides: if not provided, use tick rule (price > last price = buy, etc.)
5. Compute derived fields: trade sign (+1/-1), log returns, inter-trade durations
6. Quality checks: missing data, timestamp gaps, obvious outliers
7. Save processed data in a consistent format (Parquet for efficiency)

### Data Volume Estimate
- BTC-USDT perps on Binance: roughly 1-3 million trades per day
- 1 week across 3 venues: approximately 20-60 million trades total
- This is manageable on a laptop, especially if we process venue-by-venue

---

## 3b. Analysis Framework — Two Distinct Research Questions

This project investigates two separate but related questions about information flow in crypto markets. They use the same entropy toolkit but operate on different data groupings and answer fundamentally different things. It is important to keep them cleanly separated.

### Analysis A: Cross-Venue Perpetual Futures (Primary)

**Question:** Among the three major BTC perpetual futures venues (Binance, OKX, Bybit), where does informed trading originate and how does information propagate between them?

**Data:** Binance BTCUSDT Perp + OKX BTCUSDT Perp + Bybit BTCUSDT Perp

**Why this is apples-to-apples:** All three are the same instrument type (perpetual futures) on the same underlying (BTC) with the same quote currency (USDT). Differences in entropy, trade sign persistence, and transfer entropy between them genuinely reflect differences in trader composition, latency, and information content, not differences in instrument mechanics.

**What the entropy measures tell us here:**
- **Shannon entropy per venue:** Which venue has the most "informed" (low-entropy, directional) order flow vs the most "noisy" (high-entropy, balanced) flow?
- **Transfer entropy (3x3 matrix):** For each directed venue pair, how much does knowing one venue's recent trades reduce uncertainty about the other's near future? This reveals the information leadership hierarchy.
- **Rolling transfer entropy:** Does the leadership hierarchy shift during the crash? Does a different venue lead during high-volatility vs low-volatility regimes?

**Trading implication:** A cross-venue HFT desk can monitor transfer entropy in real time. When TE(Binance → OKX) spikes, it signals that Binance is leading and the desk should execute on OKX before the information is fully absorbed. When leadership flips, the strategy flips with it.

### Analysis B: Perpetual Futures vs Spot (Secondary / Extension)

**Question:** Does price discovery happen in the derivatives market or the spot market, and does that leadership relationship change during regime shifts?

**Data:** Binance BTCUSDT Perp (representative of the perp market) vs Coinbase BTC-USD Spot (institutional reference for spot)

**Why this requires separate treatment:** Perps and spot are fundamentally different instruments. Perps have leverage, funding rates, and attract a different trader population. Spot is unleveraged, involves actual asset transfer, and is the reference for institutional pricing (ETF NAVs, CME reference rates). Comparing them is valuable, but the interpretation of entropy differences must account for these structural differences rather than attributing everything to "information flow."

**What the entropy measures tell us here:**
- **Transfer entropy (perp → spot vs spot → perp):** Does informed trading express first in leveraged perps and then get reflected in spot? Or do institutional spot flows lead?
- **Regime-dependent leadership:** During the crash week, did the direction of information flow between perp and spot change? Liquidation cascades in perps might temporarily make perps the leader; institutional selling on spot might lead during calmer periods.

**Trading implication:** Understanding whether perp leads spot (or vice versa) tells an HFT desk where to place its monitoring infrastructure and where to execute. If perps consistently lead by 500ms, a desk watching spot is always late.

### Why the Separation Matters

Mixing perp and spot venues in a single transfer entropy matrix would conflate two effects: venue-level information advantages (what we want to measure) and instrument-level structural differences (a confound). By analysing perp-to-perp and perp-vs-spot separately, each finding has a clean interpretation.

Analysis A is the core of this project. Analysis B is a compelling extension if time permits, and the data is being collected now to keep the option open.

---

## 4. Detailed Phase Plan

### Phase 1: Setup & Data Acquisition
**Goal:** Get clean, aligned trade data from 3 venues ready for analysis.

**Tasks:**
- [x] Create GitHub repo with the structure above
- [x] Set up Python environment (requirements.txt)
- [x] Write data download scripts/instructions for each venue
- [x] Download 1-2 weeks of BTC-USDT perp trade data from Binance & Bybit *(OKX CDN archives ~6 months behind; see Note_to_Assessor.md)*
- [x] Standardise and clean data (src/data.py)
- [x] Classify trade sides where needed
- [x] Align timestamps across venues
- [x] Run quality checks and document any issues
- [x] Save processed data to data/processed/

**Key dependencies:** numpy, pandas, requests, pyarrow (for Parquet)

**Output:** Notebook 01 showing data acquisition, cleaning steps, and basic summary stats (trade counts, time coverage, data quality report).

---

### Phase 2: Exploratory Microstructure Analysis
**Goal:** Establish baseline understanding of each venue's microstructure properties before applying physics frameworks.

**Tasks:**
- [x] Compute basic statistics per venue:
  - Trade arrival rates (trades per second, by hour of day)
  - Trade size distributions (mean, median, heavy-tail check)
  - Cross-venue spread proxy (|Binance − Bybit| at 1s resolution)
- [x] Autocorrelation of trade signs at each venue
  - Plot ACF of trade signs up to 100 lags
  - Compare persistence across venues (Binance: 93 trades, Bybit: 52 trades to 1/e)
- [x] Cross-venue return correlation at various lags
  - Compute returns at 1s, 5s, 10s, 30s, 1min frequencies
  - Cross-correlation between venue pairs at each frequency
  - Lead-lag analysis via lagged cross-correlation (±30s)
- [x] Initial visualisations:
  - Intraday patterns (trade intensity by hour)
  - Price overlay across venues
  - Trade sign autocorrelation comparison
  - Correlation vs frequency plot
  - Heavy-tail CCDF (log-log)
  - Cross-venue spread vs realised volatility

**Key findings:**
- Binance ~2× Bybit in trade arrival rate (115 vs 58 trades/s mean)
- Binance has higher trade sign persistence (93 vs 52 trades to 1/e threshold)
- Cross-venue correlation: 0.93 at 1s, converging to ~1.0 at 1min (Epps effect)
- Mean cross-venue spread: $5.43, co-moves with realised volatility
- Both venues exhibit heavy-tailed trade size distributions (kurtosis: 1.1M Binance, 44K Bybit)

**Output:** Notebook 02 with exploratory findings, 6 saved figures, and trading implications per section.

---

### Phase 3: Entropy Analysis — The Core Physics
**Goal:** Apply information-theoretic measures to quantify randomness, information content, and directional information flow.

**Tasks:**

#### 3a: Shannon Entropy of Trade Signs
- [ ] Compute Shannon entropy of trade sign distributions in rolling windows
  - Window sizes: 1min, 5min, 15min (compare sensitivity)
  - H = -Σ p(x) log₂ p(x) where x ∈ {buy, sell}
  - Maximum entropy = 1 bit (50/50 split), minimum = 0 (all one side)
- [ ] Normalised entropy: H/H_max to get a 0-1 scale
- [ ] Plot entropy time series at each venue overlaid on price
- [ ] Identify periods of low entropy (informed directional trading) and high entropy (random/balanced flow)
- [ ] Compare entropy dynamics across venues — do they move together or independently?

**Trading implication:** "When entropy at venue X drops sharply (from 0.95 to 0.7 within 5 minutes), it signals a burst of informed directional trading. This preceded a 0.3% price move within the next 2 minutes in Y% of cases."

#### 3b: Transfer Entropy (Directional Information Flow)
- [ ] Implement transfer entropy: TE(X→Y) measures how much knowing X's past reduces uncertainty about Y's future
  - TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
  - Use discrete trade sign sequences binned at regular intervals (e.g., 1-second bins: net buy = +1, net sell = -1, balanced = 0)
- [ ] Compute TE for all venue pairs in both directions:
  - Binance → Bybit, Bybit → Binance
  - Binance → OKX, OKX → Binance
  - Bybit → OKX, OKX → Bybit
- [ ] Rolling transfer entropy to see how leadership shifts over time
  - Window: 30min or 1hr, stepped at 5min intervals
- [ ] Net information flow: TE(X→Y) - TE(Y→X) shows which venue is net leader
- [ ] Visualise as a time-varying heatmap

**Trading implication:** "Transfer entropy from Binance to Bybit averages X bits, while the reverse averages Y bits. Binance leads on average, but during [specific event], leadership reversed. A cross-venue strategy could monitor net TE and execute on the lagging venue when directional TE spikes."

#### 3c: Mutual Information (Optional Extension)
- [ ] Compute mutual information between venue pairs' trade sign sequences
  - MI(X;Y) = H(X) + H(Y) - H(X,Y)
  - Measures total shared information (non-directional)
- [ ] Compare MI at different time lags to find optimal lead-lag

**Key references for implementation:**
- Schreiber (2000) — original transfer entropy paper
- Bossomaier et al. (2016) — "An Introduction to Transfer Entropy" (textbook)
- Python libraries: `pyinform` or custom implementation (might be better for understanding + control)

**Output:** Notebook 03 — the centrepiece notebook. Should be the most visually rich and analytically deep.

---

### Phase 4: Phase Transition Detection
**Goal:** Look for signatures of phase transitions in market dynamics, using the entropy measures from Phase 3 as observables.

**Tasks:**

#### 4a: Temperature and Order Parameter Analogues
- [ ] Define analogues:
  - **Temperature analogue:** Realised volatility (computed from trade prices in rolling windows). High volatility = high temperature = disordered state.
  - **Order parameter analogue:** Net order flow imbalance (like magnetisation). When all trades are buys, the "market" is fully "magnetised." When balanced, it is in a disordered state.
  - **Susceptibility analogue:** How sensitive is the order flow imbalance to small perturbations? Compute variance of imbalance — it should peak near critical points.
- [ ] Plot these observables over time and identify candidate transition points

#### 4b: Correlation Length Analysis
- [ ] Compute autocorrelation of returns at various lags
- [ ] Define "correlation length" as the lag at which ACF drops below a threshold (e.g., 1/e)
- [ ] Track correlation length over time in rolling windows
- [ ] Near phase transitions (regime shifts), correlation length should diverge (increase sharply) — this is "critical slowing down"
- [ ] Test: do correlation length spikes precede major volatility regime changes?

#### 4c: Entropy Discontinuities
- [ ] Look for sharp jumps in the Shannon entropy time series from Phase 3
- [ ] Characterise: are these first-order-like (sudden jumps) or second-order-like (continuous but with diverging derivative)?
- [ ] Correlate with known market events (liquidation cascades, news, funding rate resets)

#### 4d: Regime Classification
- [ ] Using the phase transition framework, classify market periods into regimes:
  - "Hot" regime: high entropy, high volatility, low correlation length (disordered, random)
  - "Cold" regime: low entropy, low volatility, high correlation length (ordered, trending)
  - "Critical" regime: intermediate entropy, diverging correlation length, high susceptibility (transition zone)
- [ ] Validate: do these regimes correspond to intuitively different market conditions?

**Trading implication:** "When the market enters a 'critical' regime (correlation length diverging, susceptibility peaking), it signals an impending regime transition. An HFT desk could reduce position sizes or widen execution thresholds during these unstable periods, and increase aggression once the new regime is established."

**Output:** Notebook 04 — the most novel and physics-heavy notebook.

---

### Phase 5: Metastability Analysis
**Goal:** Identify quasi-stable price levels where the market lingers before transitioning, analogous to metastable states in physics.

**Tasks:**

#### 5a: Free-Energy Landscape Construction
- [ ] Construct a probability density of prices (or returns) over rolling windows
- [ ] Define free-energy analogue: F(x) = -kT ln(P(x))
  - Where P(x) is the empirical probability density at price level x
  - Local minima in F(x) = metastable states (price levels the market "likes")
  - Local maxima (barriers) = price levels the market avoids or passes through quickly
- [ ] Track how the free-energy landscape evolves over time
- [ ] Visualise as a 2D heatmap: time on x-axis, price on y-axis, colour = free energy (or probability density)

#### 5b: Dwell Time Analysis
- [ ] For identified metastable levels, compute how long price remains within a band around each level
- [ ] Compare with exponential distribution (Kramers escape theory: escape time ~ exp(barrier height / temperature))
- [ ] Does higher "temperature" (volatility) lead to faster escape from metastable levels? (It should.)

#### 5c: Connection to Support/Resistance
- [ ] Compare physics-identified metastable levels with traditional support/resistance levels
- [ ] Are they the same? Does the physics framework find levels that traditional methods miss?

**Trading implication:** "Metastable levels identified by the free-energy landscape correspond to regions where limit order density is high and price reverts. An HFT market-maker could use these levels to place resting orders with higher confidence. When the 'barrier height' decreases (free-energy landscape flattening), it signals the metastable level is weakening and a breakout is more likely."

**Output:** Notebook 05 — visually compelling with the free-energy landscape plots.

---

### Phase 6: Synthesis & Report
**Goal:** Tie everything together into a coherent narrative with clear trading implications.

**Tasks:**
- [ ] Notebook 06: Synthesis notebook that summarises key findings from all phases
  - Cross-reference: do entropy events (Phase 3) coincide with phase transitions (Phase 4) and metastability breakdowns (Phase 5)?
  - Construct a unified "market state dashboard" concept combining entropy, temperature, correlation length, and metastability metrics
  - Discuss limitations honestly (data period, simplifying assumptions, etc.)
  - Suggest concrete next steps (what would you do with 6 months and internal data?)
- [ ] Written report (report/report.md):
  - 3-5 pages, concise, well-structured
  - Abstract, Motivation, Methodology, Key Findings, Trading Implications, Limitations & Next Steps
  - Key figures embedded
- [ ] Polish README with actual results and findings
- [ ] Final code cleanup and documentation
- [ ] Ensure all notebooks run cleanly top-to-bottom
- [ ] Final git push with clean history

---

## 5. Technical Stack

### Core Dependencies
```
numpy>=1.24
pandas>=2.0
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
pyarrow>=13.0          # Parquet support
requests>=2.31         # Data downloads
tqdm>=4.65             # Progress bars
```

### Optional / As Needed
```
pyinform>=0.2          # Transfer entropy (or we implement our own)
statsmodels>=0.14      # ACF, statistical tests
scikit-learn>=1.3      # KDE, clustering if needed
plotly>=5.15           # Interactive plots (optional)
jupyterlab>=4.0        # Notebook environment
```

### Performance Note
With 20-60M trades, some computations (especially transfer entropy with rolling windows) may be slow in pure Python. Options:
- Vectorise aggressively with NumPy
- Use numba for JIT compilation of inner loops if needed
- Subsample if necessary for exploratory work, full data for final results

---

## 6. Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase transitions not cleanly visible in data | Medium | High | Report honestly; partial evidence is still interesting. Frame as "signatures consistent with phase-transition-like behaviour" rather than claiming definitive proof. |
| Transfer entropy too noisy or computationally slow | Medium | Medium | Start with simple Shannon entropy (fast, reliable). TE is an extension — if it works, great; if not, Shannon entropy + cross-correlation is still a solid story. |
| OKX data hard to obtain | Low-Medium | Low | We have Binance and Bybit as reliable fallbacks. Two venues is sufficient; three is ideal. |
| Scope creep (trying to do too much) | Medium | Medium | Stick to the phase plan. Metastability (Phase 5) is the first thing to cut if time is tight — Phases 1-4 are the core. |
| Notebooks become messy | Low | Medium | Keep analysis code in src/, use notebooks for narrative + visualisation only. Clean as you go. |
| Results are underwhelming | Low-Medium | Medium | Even null results are valuable if presented with scientific maturity. "We tested X, found limited evidence for Y, which suggests Z" is a perfectly valid quant research outcome. |

---

## 7. What "Good" Looks Like

For Mike to be impressed, the submission should demonstrate:

1. **Scientific rigour** — proper methodology, honest reporting, statistical awareness
2. **Practical grounding** — every finding linked to a trading implication
3. **Technical competence** — clean code, well-structured repo, reproducible results
4. **Clear communication** — notebooks that tell a story, not just dump outputs
5. **Physics thinking** — showing that your degree gives you a genuinely different lens on markets
6. **Self-awareness** — acknowledging limitations and suggesting next steps shows maturity

---

## 8. Where We Begin

**Step 1:** Set up the GitHub repo with the structure above.  
**Step 2:** Set up the Python environment and install dependencies.  
**Step 3:** Start downloading data from Binance (most reliable source, start there).  
**Step 4:** Build the data processing pipeline (src/data.py) and Notebook 01.  

Let's get the foundations right and the data flowing. Everything builds from there.
