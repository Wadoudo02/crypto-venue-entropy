# Market State Assessment — 2026-02-02T08:15:00Z

**Regime:** transitional  
**Confidence:** 35%  
**Risk Level:** medium  
**Crash Type:** N/A

## Reasoning

The current state presents a genuinely mixed picture that resists clean classification. On the stress side: Binance entropy has declined to the 18th percentile (H=0.92), indicating order flow is becoming more concentrated than roughly 82% of historical windows — a meaningful but sub-crisis level of directional pressure. Transfer entropy leadership has been sustained on Binance for 3 consecutive windows, which by the system's own rules constitutes confirmed leadership. Order imbalance is -0.30, reflecting net selling. On the calm side: the integrated autocorrelation time (1.1) sits below its 20-window trailing median (1.4) at the 28th percentile, the opposite of what is expected during a genuine regime transition. Critical slowing down — the ACF signature of an approaching phase transition — is absent. Susceptibility (0.004) is low, meaning the market is not near a tipping point in terms of its sensitivity to perturbation. Realised volatility (0.008) is moderate and well below the 0.015 crisis disambiguation threshold. The free-energy well at 82000 (depth 4.2) provides moderate but not strong support. The most coherent interpretation is that Binance is experiencing an early-stage increase in directional order flow, possibly from informed participants beginning to position, but the broader market structure has not yet responded — correlation lengths are short, price impact is minimal, and support levels are intact. This is a transitional state in the truest sense: some features are trending toward crisis_information but the full signature (entropy collapse + ACF spike + well depth degradation) has not converged. The correct response is heightened monitoring rather than aggressive repositioning.

## Physics Summary

The system is in a partially ordered state, analogous to a magnetic material held slightly above its Curie temperature. Local clusters of aligned spins (directional order flow on Binance) are forming, as evidenced by the entropy decline, but long-range order has not emerged — correlation lengths (ACF time) remain short and the susceptibility peak that accompanies a true phase transition is absent. The free-energy landscape retains meaningful well structure (depth 4.2) but is shallower than in a fully stable state. The system is drifting toward a potential phase transition without having committed to one. External forcing (informed selling, as suggested by order imbalance) is present but has not yet overcome the market's equilibrating forces.

## Key Features Driving Assessment

- binance_entropy_percentile: 18th — below normal, directional pressure building, but well above the 5th percentile crisis threshold
- te_leader_consecutive_windows: 3 — confirmed Binance TE leadership, but no commensurate price impact or ACF response
- acf_time: 1.1 vs trailing median 1.4 — BELOW median, critical slowing down is absent, inconsistent with imminent phase transition
- susceptibility: 0.004 — low, market is not near a tipping point
- realised_vol_30m: 0.008 — moderate, below the 0.015 high-volatility override threshold
- nearest_well_depth: 4.2 — borderline support, neither strong (>5.0) nor fragile (<2.0)

## Conflicting Signals

- Entropy at 18th percentile suggests growing directional conviction, but ACF time is BELOW its trailing median (1.1 vs 1.4) — critical slowing down is absent, which is inconsistent with a genuine regime shift
- TE leadership sustained for 3 consecutive windows (confirmed Binance dominance) yet realised volatility is only 0.008 — information flow leadership has not translated into price impact, an unusual divergence
- Order imbalance is moderately elevated at -0.30 (net selling pressure), but susceptibility is very low (0.004) — the market is absorbing sell-side flow without amplifying it, suggesting the imbalance is not yet destabilising
- Well depth of 4.2 at nearest metastable level (82000) is borderline — below the 5.0 strong-support threshold but above the 2.0 fragile-support threshold, providing no clear signal on support durability
- Binance entropy (18th percentile) and Bybit entropy (not in crisis territory) are diverging slightly — cross-venue entropy divergence can precede information-driven moves but is not yet extreme

## Confidence Rationale

Entropy has dropped to the 18th percentile, signalling emerging directional pressure, and TE leadership is sustained across 3 consecutive windows — both pointing toward a developing regime shift. However, ACF time remains below its trailing median (1.1 vs 1.4), ruling out critical slowing down, and susceptibility is low (0.004), indicating the market is not near a tipping point. Volatility is moderate (0.008), well below the crisis threshold of 0.015. The full crisis signature has not materialised. Features are genuinely split between calm-leaning (ACF, susceptibility, volatility) and stress-leaning (entropy, TE leadership, order imbalance), warranting low confidence and a transitional classification.

## Recommended Actions

- Do NOT increase position sizing — the ambiguity in the physics does not justify adding risk
- Maintain cautious passive limit orders at metastable level 82000, but be prepared to pull them quickly (well depth 4.2 is borderline; any further entropy decline or ACF spike should trigger immediate withdrawal)
- Shift execution monitoring to Binance as the confirmed TE leader (3 consecutive windows) — route aggressive orders through Binance if acting directionally
- Set alert thresholds: if binance_entropy_percentile falls below 5th AND acf_time rises above 2.8 (2x trailing median of 1.4), reclassify immediately as crisis_information and reduce passive exposure
- If realised_vol_30m rises above 0.015 with entropy remaining normal, reclassify as crisis_mechanical and shift to liquidity-provision with tight stops
- Monitor order imbalance: if it deepens below -0.45 in conjunction with entropy decline, treat as an escalating information-driven signal
