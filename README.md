## Momentum 63d Strategy (ETH)

This README explains the Momentum 63d strategy added to the backtester and how to use it.

### What is Momentum 63d?

Momentum 63d is a simple trend-following rule:

- **Go long ETH** when the trailing 63‑day return is positive.
- **Stay in cash** when the trailing 63‑day return is negative or undefined.

This is a classic absolute momentum (time‑series momentum) filter. 63 trading days ≈ 3 months.

### Exact Rule

- Define the 63‑day return as: \(R_{63}(t) = \frac{P_t}{P_{t-63}} - 1\) where \(P_t\) is today’s close.
- Position \(\in \{0, 1\}\):
  - If \(R_{63}(t) > 0\) → position = 1 (fully invested in ETH)
  - Else → position = 0 (100% cash)

Rebalances are evaluated daily using end‑of‑day prices. Trades pay the configured fee.

### Why 63 days?

- 63 trading days ≈ one quarter; a common horizon that balances responsiveness and noise.
- Empirically robust for many assets; reduces deep drawdowns during sustained downtrends.

### Pros and Cons

- Pros
  - Cuts exposure in downtrends → typically smaller drawdowns vs buy & hold
  - Very simple and transparent; few parameters to overfit
  - Low turnover (trades only at regime changes)
- Cons
  - Can whipsaw in choppy markets (small losses around the SMA/momentum line)
  - May underperform buy & hold in strong, uninterrupted bull markets

### How it’s implemented here

In the script, the position series is built as:

```python
import pandas as pd

def positions_momentum(prices: pd.Series, lookback: int) -> pd.Series:
    mom = prices.pct_change(lookback)
    pos = (mom > 0).astype(int)
    pos = pos.where(mom.notna(), 0)
    return pos
```

The backtester interprets `1` as fully long ETH and `0` as fully in cash, applying a 0.1% fee (`FEE = 0.001`) on each buy/sell.

### How to run it

1. Ensure your data files exist: `eth_last5y.json` (ETH daily prices) and optionally `btc_last5y.json`.
2. Run the script:
   ```bash
   python3 eth-strategy.py
   ```
3. Look for the section:
   - "Alternative Strategies - Filtered Window" (your `START_DATE`/`END_DATE`)
   - "Alternative Strategies - Full History"
   The row named `Momentum 63d` shows final value, percent return, and out/under‑performance vs buy & hold.

To change the date range, set `START_DATE` and `END_DATE` at the top of `eth-strategy.py`.

### Backtest highlights on your data

- 2025 window (your current `START_DATE`–`END_DATE`): Momentum 63d outperformed buy & hold in our run (approx. +112.2% vs +40.9%).
- 2020–2025 full history: Momentum 63d produced large gains but still trailed raw buy & hold over the entire bull run. It typically shines in sideways/bear regimes by reducing drawdowns.

Results will change with the evaluation window, fees, and precise data. Use multiple windows for a fair assessment.

### Tuning

- Lookback: try 42d/84d/126d to trade off responsiveness vs stability.
- Optional enhancements (not required):
  - Volatility targeting: scale the position to a volatility budget instead of 0/1.
  - Ensemble: hold ETH if either Momentum 63d OR a long‑term SMA filter is bullish.

### Risk Notes

- Momentum can suffer from whipsaws during choppy periods.
- There is no guarantee of future outperformance; use as one tool in a diversified process.

### License / Usage

This repository is provided for research and educational purposes only. No financial advice.