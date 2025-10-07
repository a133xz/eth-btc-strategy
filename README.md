# ETH-BTC Trading Strategy

This is a **systematic trading strategy** designed to capture Bitcoin price movements using **asymmetric thresholds**:

| Action   | Threshold | Purpose                        |
| -------- | --------- | ------------------------------ |
| **Sell** | +7%       | Take profits on upward moves   |
| **Buy**  | -1%       | Re-enter after small pullbacks |

**How it works:**

1. **Buy BTC** with available capital (initial investment or reinvested profits).
2. **Sell BTC** when the price rises 7% from the entry.
3. **Reinvest fully** all proceeds for the next purchase.

### Example Trade Sequence

| Day | BTC Price | Action | BTC Holdings | Capital |
| --- | --------- | ------ | ------------ | ------- |
| 1   | 10,000    | Buy    | 0.5 BTC      | $5,000  |
| 5   | 10,700    | Sell   | 0 BTC        | $5,350  |
| 8   | 10,665    | Buy    | 0.501 BTC    | $5,350  |
| 12  | 11,372    | Sell   | 0 BTC        | $5,697  |

### Why It Works

**Asymmetric Positioning:**

* **Sell at +7%** → larger profit-taking, accumulates more BTC at lower prices
* **Buy at -1%** → smaller re-entry, reduces risk exposure

**Result:** More BTC is accumulated in uptrends, and losses are minimized in pullbacks, creating compounding growth.

1. **Volatility = Opportunity:** Bitcoin price swings create systematic profit chances.
2. **Asymmetric Trading:** Larger profit threshold than loss tolerance.
3. **Compounding:** Full reinvestment drives exponential growth.
4. **Risk Management:** Thresholds limit drawdowns.

## Configuration

```python
# Trading parameters
SELL_THRESHOLD = 0.07   # Sell at +7% profit
BUY_THRESHOLD  = 0.01   # Buy at -1% loss
FEE            = 0.001  # 0.1% trading fee

# Investment
INITIAL_USD = 5000      # Starting capital

# Backtest date range
START_DATE = "2020-09-08"
END_DATE   = "2025-10-07"
```