# Note to Assessor — Data Source Availability

## What We Aimed For

The original plan was to acquire trade-level data from four venues for the same 7-day period (2026-01-30 to 2026-02-05):

1. **Binance** — BTCUSDT Perpetual Futures (perp-to-perp analysis)
2. **OKX** — BTC-USDT-SWAP Perpetual Futures (perp-to-perp analysis)
3. **Bybit** — BTCUSDT Perpetual Futures (perp-to-perp analysis)
4. **Coinbase** — BTC-USD Spot (perp-vs-spot analysis, separate from the core perp comparison)

## What We Got

Only **Binance** and **Bybit** provide bulk historical trade downloads for recent dates. The other venues were investigated and the download functions were implemented, but data retrieval was not practical within the project timeline.

## Why Each Venue Was Unavailable

### OKX

- **Bulk download** (`https://www.okx.com/cdn/okex/traderecords/trades/monthly/{YYYYMM}/allfuture-trades-{YYYY-MM-DD}.zip`): These monthly archives are published with a **~6-month delay**. Testing confirmed data is available up to August 2025 but returns HTTP 404 for any month from September 2025 onward. Our target dates (Jan-Feb 2026) are too recent.
- **REST API** (`/api/v5/market/history-trades`): Returns only **100 trades per request** with no timestamp-based filtering — pagination is purely by trade ID, going backwards from the most recent trade. Reaching data from 8 days ago would require millions of sequential API calls, which is impractical.
- **Code written:** `download_okx_trades()` and `_download_okx_rest()` in `src/data.py` — fully functional, will work once CDN archives are published for our date range.

### Coinbase (BTC-USD Spot)

- **No bulk download** exists. Coinbase only offers a REST API.
- **REST API** (`GET /products/BTC-USD/trades`): Returns **1,000 trades per page** with `cb-after` cursor pagination. Effective throughput: ~1,500 trades/sec. Covering 8+ days of data (~5M trades) requires ~55 minutes of continuous API calls. This exceeded the notebook execution timeout (2 hours total including all other processing).
- **Note:** Coinbase spot was intended for a secondary analysis (perp-vs-spot information flow), not the core perp-to-perp transfer entropy study. Its absence does not affect the primary research question.
- **Code written:** `download_coinbase_trades()` and `_save_coinbase_daily()` in `src/data.py` — fully functional, tested and confirmed working. Could be run as a standalone pre-download step with sufficient time.

### Gate.io (Explored as Alternative)

- **No bulk download** available for futures trade data.
- **REST API** (`/api/v4/futures/usdt/trades`): Returns **1,000 trades per page** at ~526 trades/sec effective rate. Covering 9 days would require approximately **5 hours** of continuous pagination.
- **Not implemented** in code — ruled out after API testing showed impractical download times.

## What This Means for the Analysis

The core research question — **where does informed trading originate and how does information flow between perpetual futures venues?** — is fully addressable with Binance + Bybit:

- Both are major BTCUSDT perpetual futures venues with the same instrument mechanics
- They have different trader compositions, latency profiles, and market share
- Transfer entropy TE(Binance → Bybit) vs TE(Bybit → Binance) directly reveals the information leadership hierarchy
- Rolling TE captures how that hierarchy shifts during regime changes

A third perp venue (OKX) would enrich the analysis with a 3×3 transfer entropy matrix, but the 2×2 matrix from Binance + Bybit is sufficient for a compelling demonstration of the methodology.

## If More Time Were Available

1. **Run Coinbase download overnight** as a standalone script, then load cached CSVs in the notebook
2. **Wait for OKX CDN archives** to be published for Jan-Feb 2026 (expected ~July 2026)
3. **Use a data vendor** (e.g. Tardis.dev, Kaiko) that provides pre-aggregated historical trade data via API with timestamp-based queries
