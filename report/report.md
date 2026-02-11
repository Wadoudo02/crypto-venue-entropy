# Statistical Mechanics of Cross-Venue Information Flow in Bitcoin Perpetual Futures

## Abstract

This study applies statistical mechanics to quantify cross-venue information flow in Bitcoin perpetual futures during a major crash period (Jan 30 to Feb 6, 2026, 84K to 62K) across Binance and Bybit. Using Shannon entropy, transfer entropy, phase transition detection, and free-energy landscape construction, we identify three operationally distinct market states and demonstrate that two crashes within the same week exhibit fundamentally different microstructural signatures. The integrated autocorrelation time ($\tau_{\mathrm{int}}$) provides a genuine forward-looking signal ($\rho = 0.34$ with 30-minute forward volatility), Shannon entropy below the 5th percentile precedes significant price moves 88.1% of the time, and physics-based metastable levels overlap 90% with traditional support/resistance while adding quantitative strength measures. The central finding is that information-driven crashes (low entropy, clear venue leadership) and mechanically-driven crashes (normal entropy, liquidation cascades) require fundamentally different trading responses, and the statistical mechanics framework distinguishes them in real time.

## 1. Motivation

Cross-venue price discovery in crypto perpetual futures markets is poorly understood at the microstructure level. With BTC perpetual futures trading simultaneously on Binance, Bybit, OKX, and other venues, information does not originate uniformly: some venues lead, some follow, and the leadership hierarchy shifts dynamically. For a cross-venue HFT desk, three questions are critical: (1) where does informed trading originate, (2) when does the market undergo regime shifts, and (3) what are the quasi-stable price levels where the market lingers before transitioning?

Traditional approaches to these questions use linear correlation, simple volatility thresholds, and heuristic support/resistance (round numbers, swing points). This project tests whether statistical mechanics provides a more principled framework. We map market observables to thermodynamic quantities: trade sign entropy measures disorder, transfer entropy measures directional information flow, realised volatility serves as temperature, and the empirical price distribution defines a free-energy landscape with metastable states. The question is not whether markets are literally physical systems (they are not), but whether the mathematical tools of statistical mechanics extract useful structure that traditional methods miss.


### Physics Analogies: A Non-Physicist's Guide

The statistical mechanics framework maps market observables onto physical quantities. The mapping is not literal (markets are not physical systems), but the mathematical tools built for physics extract structure that traditional financial metrics miss. Here is the intuition behind each concept.

**Entropy** measures disorder. Imagine shuffling a deck of cards: a perfectly alternating red-black sequence has low entropy (highly ordered), while a random shuffle has high entropy. In our context, entropy measures the disorder of the buy/sell trade sequence. When entropy is high (~1.0), buys and sells arrive in no particular pattern, meaning no single group is dominating the order flow. When entropy drops sharply, one side is trading with conviction: the sequence becomes ordered, like a run of consecutive red cards. That ordering is the fingerprint of informed directional trading.

**Transfer entropy** measures who is copying whom. Standard correlation tells you two things move together; transfer entropy tells you which one moves *first* in a statistically meaningful way. If knowing Binance's recent trades helps you predict Bybit's next trade (beyond what Bybit's own history tells you), information is flowing from Binance to Bybit. It is the difference between "these two venues are correlated" and "Binance is leading."

**Temperature** is realised volatility. In physics, temperature measures how energetically particles are bouncing around. Here, it measures how energetically price is moving. A "hot" market has large, frequent price changes; a "cold" market is quiet. The analogy is direct: just as heating a solid causes its atoms to vibrate more violently, increased volatility causes price to fluctuate more aggressively around any given level.

**Free-energy landscape** is perhaps the most useful analogy. Picture a ball rolling on a hilly surface. The valleys are where the ball naturally settles (stable price levels), and the hills between them are barriers the ball must overcome to move to the next valley. We construct this landscape empirically: prices the market visits frequently become valleys (low free energy), and prices the market avoids become hills (high free energy). A deep valley means the market keeps returning to that price; a shallow valley means it lingers only briefly before moving on. This is the physics version of support and resistance, but with a continuous, quantitative measure of *how strong* each level is.

**Metastable states** are the shallow valleys. A ball sitting in a shallow bowl is stable for now, but a small nudge will push it over the rim. A ball in a deep bowl needs a large push. Metastable price levels are the same: the market sits there temporarily, but the level will eventually break. The depth of the well tells you how much force (trading pressure) is needed to break through. Traditional support/resistance is binary (level exists or it does not); the free-energy framework gives you a scalar strength measure and lets you watch it degrade in real time.

**Phase transitions** are regime shifts. Water does not gradually become ice; it undergoes an abrupt transition at a specific temperature. Markets exhibit analogous behaviour: a quiet, mean-reverting regime can shift abruptly into a trending, volatile regime. The physics framework provides tools to detect when the system is *approaching* such a transition, not just that one has already occurred.

**Critical slowing down** ($\tau_{\mathrm{int}}$) is the early warning. When a physical system approaches a phase transition, it takes longer to return to equilibrium after a perturbation. Poke a glass of water and it settles quickly; poke water at 99°C and the fluctuations persist longer. The integrated autocorrelation time measures this: when $\tau_{\mathrm{int}}$ rises, the market's volatility structure is becoming self-reinforcing, and perturbations take longer to dissipate. Empirically, this precedes volatility spikes ($\rho = 0.34$), making it a genuine forward-looking warning signal.

**Kramers escape theory** predicts how long a ball stays in a valley before thermal fluctuations push it over the barrier. The prediction is exponential: double the barrier height, and the expected dwell time increases exponentially. We tested this against real dwell times at metastable price levels and found only a weak relationship ($\rho = 0.157$). The reason is intuitive: in physics, escape is driven by random thermal noise. In markets, escape is driven by liquidation cascades and informed order flow, which are far more violent than "thermal" fluctuations. The market does not gently wander out of a support level; it gets shoved.

**The punchline:** Each physics tool targets a specific trading question. Entropy tells you *who is trading with conviction*. Transfer entropy tells you *which venue knows first*. Temperature tells you *how volatile conditions are*. The free-energy landscape tells you *where price wants to sit and how strongly*. And critical slowing down tells you *when a regime shift is approaching*. None of these require the market to actually be a physical system; they just require that the mathematical machinery, built over a century of statistical physics, is good at detecting structure in noisy data. It is.



## 2. Data and Methodology

### Data

We analyse 7 days of BTC-USDT perpetual futures trade data (Jan 30 to Feb 6, 2026) from two venues:

- **Binance:** 69.4 million trades, mean arrival rate 115 trades/s
- **Bybit:** 35.0 million trades, mean arrival rate 58 trades/s

This period includes two major crashes: a $6K drop on Jan 31 (from $84K to $78K) and a $13K drop on Feb 5-6 (from $75K to $62K), providing a natural experiment in regime transitions.

### Analytical Framework

The analysis proceeds in four stages, each building on the previous:

1. **Microstructure exploration** (Phase 2): Trade arrival rates, size distributions, autocorrelation structure, and the Epps effect (cross-venue correlation vs timescale).

2. **Entropy analysis** (Phase 3): Shannon entropy of trade signs in rolling 5-minute windows quantifies order flow disorder. Transfer entropy at 1-second resolution measures directional information flow between venues. Mutual information as a function of lag quantifies the information-sharing timescale.

3. **Phase transition detection** (Phase 4): Realised volatility (temperature), order flow imbalance (order parameter), and susceptibility define a thermodynamic state space. The integrated autocorrelation time ($\tau_{\mathrm{int}} = 0.5 + \sum_{k=1}^{k^*} \mathrm{ACF}(k)$, truncated at the first negative ACF) measures critical slowing down. Entropy discontinuity detection identifies first-order-like regime transitions.

4. **Metastability analysis** (Phase 5): The free-energy landscape $F(x) = -k_BT \ln P(x)$ is constructed from empirical price distributions in rolling 4-hour windows. Local minima identify metastable levels with quantitative well depths. Dwell time analysis measures how long the market lingers at each level, and Kramers escape theory is tested against the data.

## 3. Key Findings

### 3.1 Cross-Venue Information Flow

Transfer entropy resolves an information leadership hierarchy invisible to linear cross-correlation, which showed no detectable lead-lag at 1-second resolution (Phase 2). At history length $k=1$, TE(Binance $\to$ Bybit) = 0.00674 bits exceeds TE(Bybit $\to$ Binance) = 0.00641 bits, with Binance leading in 59.4% of rolling 30-minute windows. Transfer entropy is statistically significant (vs shuffled baseline) in 53.2% of Binance-led windows.

An important caveat: at $k=2$ and $k=3$, the leadership reverses to Bybit. This suggests Binance has faster "first-mover" influence (its most recent action predicts Bybit's next), while Bybit's influence operates through longer-range temporal patterns. The $k=1$ result is more relevant for low-latency execution; $k=2$-$3$ may be relevant for longer-horizon strategies.

Mutual information peaks at lag = 0 seconds (0.076 bits) and drops 86% within 1 second, confirming that the cross-venue information-sharing timescale is sub-second. Any informational edge from observing one venue's flow must be acted upon within 1-2 seconds.

![Rolling transfer entropy showing Binance-Bybit information flow dynamics](../figures/03_rolling_transfer_entropy.png)

### 3.2 Regime Detection and Phase Transitions

The integrated autocorrelation time provides the strongest forward-looking signal in the study: $\rho = 0.34$ with 30-minute forward volatility. When $\tau_{\mathrm{int}}$ exceeds its 90th percentile, subsequent volatility is 1.65$\times$ the baseline. This is a genuine early warning: the market's volatility structure becomes self-reinforcing before the regime shift fully materialises.

Entropy discontinuity detection reveals 63 first-order-like transitions, heavily concentrated around the Jan 31 crash but largely absent during the Feb 5-6 crash. This asymmetry is the key to the "two crashes, two mechanisms" finding (Section 3.4).

Regime classification using a 2-of-3 scoring system (temperature, entropy, correlation length) assigns clear labels (Hot, Cold, or Critical) to only ~44% of windows; the remaining ~56% are classified as Transitional, reflecting the stringency of requiring two simultaneous extreme-quartile indicators and indicating that markets are far from the clean phase separation of equilibrium systems.

![Correlation length evolution showing critical slowing down before crashes](../figures/04_correlation_length.png)

### 3.3 Metastable Price Levels

The free-energy landscape constructed from rolling 4-hour price distributions identifies 98 metastable levels across 162 windows, with well depths ranging from 0.8 (shallow, transient) to 7.5 (deep, persistent). These physics-based levels overlap 90% with traditional support/resistance (18 of 20 traditional levels matched within $\pm$1%).

The value-add over traditional S/R is threefold: (1) quantitative strength via well depth, distinguishing strong support (depth > 5.0) from weak (depth < 2.0); (2) temporal evolution, where monitoring depth degradation across successive windows provides early warning of level failure; and (3) 13 additional levels at non-round-number consolidation points invisible to traditional heuristics.

Dwell times at metastable levels follow an approximately exponential distribution (median 41 seconds, mean 134.5 seconds, $\lambda = 0.0074$), consistent with a memoryless escape process. Kramers escape theory predicts $\tau \sim \exp(\Delta F / k_BT)$; the empirical correlation between barrier height and log-dwell-time is weak ($\rho = 0.157$), consistent with an externally driven system where liquidation cascades override thermal escape dynamics.

![Free-energy landscape showing metastable price structure](../figures/05_free_energy_landscape.png)

### 3.4 Two Crashes, Two Mechanisms

The central narrative finding: the Jan 31 and Feb 5-6 crashes are structurally different, and the statistical mechanics framework distinguishes them cleanly.

**Jan 31 ($84K to $78K): Information-driven.**
- Shannon entropy collapses to $H = 0.59$ (well below the 5th percentile of 0.958)
- Transfer entropy spikes with clear Binance leadership
- 63 entropy discontinuities concentrate in this period
- $\tau_{\mathrm{int}}$: sharp spike to $\sim$38 lags (critical-point-like)
- Metastable levels: deep wells eroding before break

This is a crash driven by informed directional trading, with information cascading from the leading venue.

**Feb 5-6 ($75K to $62K): Mechanically-driven.**
- Entropy stays near 1.0 despite a larger absolute price decline
- Transfer entropy elevated but bidirectional, with no clear leader
- Entropy discontinuities largely absent
- $\tau_{\mathrm{int}}$: broader, lower elevation
- Metastable levels: shallow wells, rapid staircase breakdown

This is a crash driven by liquidation cascades and forced selling, where both sides of the order book are active.

![Entropy discontinuities revealing crash type asymmetry](../figures/04_entropy_discontinuities.png)

![Side-by-side crash comparison](../figures/06_crash_comparison.png)

## 4. Trading Implications

Every finding maps to a specific, quantitative trading action:

**Real-time monitoring.** Shannon entropy below the 5th percentile on Binance signals a directional burst; 88.1% of such signals preceded |return| > 0.05% within 5 minutes. This is a high-confidence trigger to reduce passive exposure.

**Venue selection.** Net transfer entropy identifies the information-leading venue in real time. When leadership reverses (net TE flips sign for 2+ consecutive windows), execution should shift to the new leader.

**Risk management.** $\tau_{\mathrm{int}}$ exceeding $2\times$ its trailing median signals regime instability ($\rho = 0.34$ with forward volatility). Reduce position sizes and widen execution bands.

**Crash-type identification.** Within a few 5-minute entropy windows, the entropy-TE signature identifies whether a crash is information-driven (low entropy, clear TE leadership; follow or fade the informed flow) or mechanically-driven (normal entropy, bidirectional TE; provide liquidity at deep metastable levels, as mean-reversion is more likely once the cascade exhausts).

**Order placement.** Metastable levels with well depth > 5.0 in stable regimes ($\tau_{\mathrm{int}}$ < median) are safe targets for passive limit orders. Median dwell time of 41 seconds defines the stale-order timeout. When well depth degrades below 2.0, pull resting bids.

**Complementary signals.** Rather than converging on the same events, entropy and metastability signals provide complementary coverage of different crash types. Low entropy preceded 17.8% of major price moves (information-driven); weak well depth preceded 9.9% (structural breakdown). Together, 27.5% of major moves were preceded by at least one signal within 30 minutes, with almost no overlap, confirming the two-crash-type finding.

## 5. Limitations and Future Work

### Limitations

- **Single crash period.** Results are conditioned on a 7-day bearish window (Jan 30 to Feb 6, 2026) and may not generalise to ranging or bullish markets.

- **Equilibrium framework applied to a non-equilibrium system.** The statistical mechanics vocabulary is useful, but quantitative predictions transfer only partially. Kramers escape theory showed only a weak positive correlation between barrier height and dwell time ($\rho = 0.157$, Phase 5), and a separate test of whether ambient regime stability ($\tau_{\mathrm{int}}$) predicts dwell time found no meaningful relationship ($\rho \approx -0.08$, Phase 6).

- **Transfer entropy is history-length sensitive.** At $k=1$, Binance leads; at $k=2$-$3$, Bybit leads. The leadership hierarchy depends on the timescale of interest.

- **Regime classification is ambiguous ~56% of the time** (Transitional), and regime transitions do not predict higher forward volatility ($0.92\times$). Regime labels are concurrent state descriptors, not forward-looking signals.

- **Resolution floor.** 1-second binning averages over sub-second dynamics. The Epps effect and MI decay both suggest information propagates faster than our resolution can capture.

### Future Directions

With additional time and data, the most promising extensions are:

- **Sub-second data** to resolve the information propagation timescale below our 1-second floor, where 86% of cross-venue MI is absorbed.
- **Cross-venue metastability** to test whether Binance levels predict Bybit support/resistance with a lag, directly connecting information leadership (Phase 3) with the free-energy framework (Phase 5).
- **Non-equilibrium escape models** (hazard rates conditioned on market state) to replace the failed Kramers framework, leveraging the exponential dwell time distribution as a starting point.
- **Live dashboard** implementing the five-panel market state framework on streaming data; all observables are computationally lightweight.
- **Formal backtesting** of a combined entropy-metastability strategy across multiple market regimes (minimum 3 months), measuring P&L, Sharpe ratio, and maximum drawdown.

## Personal Note

This project was a lot of fun! I really wanted to encorporate my background and love for Physics as well as my deep interest in Quantitative Research into one project together. I feel this project certainly gave me a glimpse into the world where this is possible. 

## References

- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461.
- Epps, T. W. (1979). Comovements in stock prices in the very short run. *Journal of the American Statistical Association*, 74(366), 291-298.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
- Kramers, H. A. (1940). Brownian motion in a field of force and the diffusion model of chemical reactions. *Physica*, 7(4), 284-304.
- Sokal, A. D. (1997). Monte Carlo methods in statistical mechanics: Foundations and new algorithms. *Functional Integration*, 131-192.
