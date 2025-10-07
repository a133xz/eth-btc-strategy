import pandas as pd
import numpy as np
import json
from itertools import product

# =====================
# CONFIGURATION
# =====================
FEE = 0.001  # 0.1% trading fee
INITIAL_USD = 5000

PRINT_DETAILED = False  # Set to False to skip printing detailed transactions
DEBUG_MODE = False  # Set to True to see detailed output

# Strategy thresholds (percentage changes)
SELL_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
BUY_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

# Date range filter
# earliest date is 2020-10-08
START_DATE = "2025-01-08"
END_DATE = "2025-10-06"

# Your chosen thresholds
MY_SELL_THRESHOLD = 0.04
MY_BUY_THRESHOLD = 0.02

# =====================
# Load BTC price data
# =====================
with open("btc_last5y.json", "r") as f:
    raw_data = json.load(f)

# Keep a full-history DataFrame and a filtered working copy
data_full = pd.DataFrame(raw_data)
data_full["date"] = pd.to_datetime(data_full["date"])
data_full = data_full.sort_values("date").reset_index(drop=True)

data = data_full.copy()
if START_DATE:
    data = data[data["date"] >= pd.to_datetime(START_DATE)]
if END_DATE:
    data = data[data["date"] <= pd.to_datetime(END_DATE)]

# =====================
# Helper indicators & backtester
# =====================
def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def positions_sma_filter(prices: pd.Series, window: int) -> pd.Series:
    sma = compute_sma(prices, window)
    pos = (prices > sma).astype(int)
    pos = pos.where(sma.notna(), 0)
    return pos

def positions_ma_crossover(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_sma = compute_sma(prices, fast)
    slow_sma = compute_sma(prices, slow)
    pos = (fast_sma > slow_sma).astype(int)
    pos = pos.where(slow_sma.notna(), 0)
    return pos

def positions_momentum(prices: pd.Series, lookback: int) -> pd.Series:
    mom = prices.pct_change(lookback)
    pos = (mom > 0).astype(int)
    pos = pos.where(mom.notna(), 0)
    return pos

def positions_donchian(prices: pd.Series, window: int) -> pd.Series:
    rolling_high = prices.rolling(window=window, min_periods=window).max()
    rolling_low = prices.rolling(window=window, min_periods=window).min()
    long_signal = prices >= rolling_high
    flat_signal = prices <= rolling_low
    pos = pd.Series(0, index=prices.index)
    in_pos = 0
    for i in range(len(prices)):
        if not np.isfinite(rolling_high.iat[i]) or not np.isfinite(rolling_low.iat[i]):
            pos.iat[i] = 0
            continue
        if in_pos == 0 and long_signal.iat[i]:
            in_pos = 1
        elif in_pos == 1 and flat_signal.iat[i]:
            in_pos = 0
        pos.iat[i] = in_pos
    return pos

def positions_eth_btc_relative(data_eth: pd.DataFrame, data_btc: pd.DataFrame, window: int) -> pd.Series:
    df = (
        data_eth[["date", "price"]]
        .rename(columns={"price": "eth"})
        .merge(
            data_btc[["date", "price"]].rename(columns={"price": "btc"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    ratio = df["eth"] / df["btc"]
    ratio_sma = ratio.rolling(window=window, min_periods=window).mean()
    pos = (ratio > ratio_sma).astype(int)
    pos = pos.where(ratio_sma.notna(), 0)
    pos.index = df.index
    df_ret = df.copy()
    df_ret["pos"] = pos
    # Re-align back to the ETH filtered index by date
    aligned = (
        data_eth[["date"]]
        .merge(df_ret[["date", "pos"]], on="date", how="left")
        .set_index(data_eth.index)
    )
    return aligned["pos"].fillna(0).astype(int)

def backtest_from_positions(df: pd.DataFrame, pos: pd.Series) -> float:
    usd = INITIAL_USD
    eth = 0.0
    last_pos = 0
    for i in range(len(df)):
        price = df.iloc[i]["price"]
        p = int(pos.iloc[i]) if i < len(pos) else 0
        if last_pos == 0 and p == 1:
            if usd > 0:
                eth_bought = (usd * (1 - FEE)) / price
                eth += eth_bought
                usd = 0.0
        elif last_pos == 1 and p == 0:
            if eth > 0:
                usd_received = eth * price * (1 - FEE)
                usd += usd_received
                eth = 0.0
        last_pos = p
    final_value = usd + eth * df.iloc[-1]["price"]
    return final_value

# =====================
# Load BTC for relative strategies
# =====================
with open("btc_last5y.json", "r") as f:
    btc_raw = json.load(f)
btc_full = pd.DataFrame(btc_raw)
btc_full["date"] = pd.to_datetime(btc_full["date"])
btc_full = btc_full.sort_values("date").reset_index(drop=True)

def run_and_report_strategies(df: pd.DataFrame, label: str):
    prices = df["price"].reset_index(drop=True)
    df_local = df.reset_index(drop=True)

    strategies = {}
    strategies["SMA200 filter"] = positions_sma_filter(prices, 200)
    strategies["SMA100 filter"] = positions_sma_filter(prices, 100)
    strategies["MA cross 50/200"] = positions_ma_crossover(prices, 50, 200)
    strategies["Momentum 63d"] = positions_momentum(prices, 63)
    strategies["Donchian 50d"] = positions_donchian(prices, 50)
    strategies["ETH/BTC RS 90d"] = positions_eth_btc_relative(df_local[["date", "price"]], btc_full, 90)

    results_alt = []
    initial_price_local = df_local.iloc[0]["price"]
    final_price_local = df_local.iloc[-1]["price"]
    hold_value_local = INITIAL_USD * (final_price_local / initial_price_local)
    hold_pct_local = ((hold_value_local - INITIAL_USD) / INITIAL_USD) * 100

    for name, pos in strategies.items():
        # Align pos length to df_local
        pos_aligned = pos
        if len(pos_aligned) != len(df_local):
            pos_aligned = pos_aligned.reindex(range(len(df_local))).fillna(0)
        value = backtest_from_positions(df_local, pos_aligned)
        pct = ((value - INITIAL_USD) / INITIAL_USD) * 100
        results_alt.append({
            "strategy": name,
            "final_value": value,
            "percent_change": pct,
            "vs_hold_pct": pct - hold_pct_local,
        })

    results_alt_df = pd.DataFrame(results_alt).sort_values("final_value", ascending=False).reset_index(drop=True)

    print(f"\nAlternative Strategies - {label}")
    print(results_alt_df.head(10))


def parameter_sweep(df: pd.DataFrame, label: str):
    prices = df["price"].reset_index(drop=True)
    df_local = df.reset_index(drop=True)
    combos = []

    # Momentum lookbacks
    for lb in [21, 42, 63, 84, 126, 252]:
        pos = positions_momentum(prices, lb)
        value = backtest_from_positions(df_local, pos)
        pct = ((value - INITIAL_USD) / INITIAL_USD) * 100
        combos.append((f"Momentum {lb}d", value, pct))

    # SMA filters
    for w in [50, 100, 150, 200, 250]:
        pos = positions_sma_filter(prices, w)
        value = backtest_from_positions(df_local, pos)
        pct = ((value - INITIAL_USD) / INITIAL_USD) * 100
        combos.append((f"SMA{w} filter", value, pct))

    # MA crossovers
    for fast in [10, 20, 50]:
        for slow in [100, 150, 200]:
            if fast >= slow:
                continue
            pos = positions_ma_crossover(prices, fast, slow)
            value = backtest_from_positions(df_local, pos)
            pct = ((value - INITIAL_USD) / INITIAL_USD) * 100
            combos.append((f"MA cross {fast}/{slow}", value, pct))

    # Donchian
    for w in [20, 50, 100, 150]:
        pos = positions_donchian(prices, w)
        value = backtest_from_positions(df_local, pos)
        pct = ((value - INITIAL_USD) / INITIAL_USD) * 100
        combos.append((f"Donchian {w}d", value, pct))

    # ETH/BTC RS
    for w in [60, 90, 120, 180]:
        pos = positions_eth_btc_relative(df_local[["date", "price"]], btc_full, w)
        value = backtest_from_positions(df_local, pos)
        pct = ((value - INITIAL_USD) / INITIAL_USD) * 100
        combos.append((f"ETH/BTC RS {w}d", value, pct))

    initial_price_local = df_local.iloc[0]["price"]
    final_price_local = df_local.iloc[-1]["price"]
    hold_value_local = INITIAL_USD * (final_price_local / initial_price_local)
    hold_pct_local = ((hold_value_local - INITIAL_USD) / INITIAL_USD) * 100

    results = [
        {"strategy": name, "final_value": val, "percent_change": pct, "vs_hold_pct": pct - hold_pct_local}
        for (name, val, pct) in combos
    ]
    df_res = pd.DataFrame(results).sort_values("final_value", ascending=False).reset_index(drop=True)
    print(f"\nParameter Sweep - {label}")
    print(df_res.head(15))

# =====================
# Simulation function
# =====================
def simulate_strategy(data, sell_thresh, buy_thresh):
    usd = INITIAL_USD
    btc = 0
    last_buy_price = None
    state = "USD"
    transactions = []

    for _, row in data.iterrows():
        date = row["date"]
        price = row["price"]

        if state == "USD" and usd > 0:
            btc_bought = (usd * (1 - FEE)) / price
            if btc_bought > 0:
                transactions.append({
                    "date": date,
                    "action": "BUY",
                    "price": price,
                    "btc_amount": btc_bought,
                    "usd_spent": usd
                })
                btc += btc_bought
                last_buy_price = price
                usd = 0
                state = "BTC"

        elif state == "BTC" and btc > 0:
            if price >= last_buy_price * (1 + sell_thresh):
                usd_received = btc * price * (1 - FEE)
                if btc > 0 and usd_received > 0:
                    transactions.append({
                        "date": date,
                        "action": "SELL",
                        "price": price,
                        "btc_amount": btc,
                        "usd_received": usd_received
                    })
                    usd = usd_received
                    btc = 0
                    last_buy_price = None
                    state = "USD"
            elif price <= last_buy_price * (1 - buy_thresh) and usd > 0:
                btc_bought = (usd * (1 - FEE)) / price
                if btc_bought > 0:
                    transactions.append({
                        "date": date,
                        "action": "BUY",
                        "price": price,
                        "btc_amount": btc_bought,
                        "usd_spent": usd
                    })
                    btc += btc_bought
                    last_buy_price = (last_buy_price + price) / 2
                    usd = 0
                    state = "BTC"

    final_value = usd + btc * data.iloc[-1]["price"]
    return final_value, transactions

# =====================
# Run simulations
# =====================
results = []
for sell_thresh, buy_thresh in product(SELL_THRESHOLDS, BUY_THRESHOLDS):
    final_value, transactions = simulate_strategy(data, sell_thresh, buy_thresh)
    results.append({
        "sell_threshold": sell_thresh,
        "buy_threshold": buy_thresh,
        "final_value": final_value,
        "transactions_count": len(transactions),
        "transactions": transactions
    })

results_df = pd.DataFrame(results)
results_df["percent_change"] = ((results_df["final_value"] - INITIAL_USD) / INITIAL_USD) * 100
results_df = results_df.sort_values(by="final_value", ascending=False).reset_index(drop=True)

# =====================
# Find your strategy
# =====================
my_row = results_df[
    (results_df["sell_threshold"] == MY_SELL_THRESHOLD) &
    (results_df["buy_threshold"] == MY_BUY_THRESHOLD)
]

# =====================
# Display results
# =====================
print("\nTop 5 Strategies Overall:")
print(results_df.head(5)[["sell_threshold", "buy_threshold", "final_value", "percent_change", "transactions_count"]])

if not my_row.empty:
    print(f"\nYour Strategy (Sell={MY_SELL_THRESHOLD}, Buy={MY_BUY_THRESHOLD}):")
    print(my_row[["sell_threshold", "buy_threshold", "final_value", "percent_change", "transactions_count"]])
    
    if PRINT_DETAILED:
        print("\nDetailed Transactions:")
        for t in my_row.iloc[0]["transactions"]:
            if t["action"] == "BUY":
                print(f"{t['date'].date()}: Bought {t['btc_amount']:.6f} BTC at ${t['price']:.2f} spending ${t['usd_spent']:.2f}")
            elif t["action"] == "SELL":
                print(f"{t['date'].date()}: Sold {t['btc_amount']:.6f} BTC at ${t['price']:.2f} receiving ${t['usd_received']:.2f}")

def simulate_strategy_profit_only(data, sell_thresh, buy_thresh, initial_usd):
    locked_capital = initial_usd  # never reinvested - represents initial investment
    profit_only = 0               # profit available to reinvest
    btc = 0
    total_cost_basis = 0         # tracks total amount invested in BTC
    state = "USD"

    for _, row in data.iterrows():
        price = row["price"]

        if state == "USD" and profit_only > 0:
            # Use profit for reinvestment
            usd_to_invest = profit_only
            btc_bought = (usd_to_invest * (1 - FEE)) / price
            btc += btc_bought
            total_cost_basis += usd_to_invest  # Track that we invested profit
            profit_only = 0
            state = "BTC"

        elif state == "USD" and btc == 0 and locked_capital > 0:
            # First trade: use initial capital
            usd_to_invest = locked_capital
            btc_bought = (usd_to_invest * (1 - FEE)) / price
            btc += btc_bought
            total_cost_basis += usd_to_invest  # Track the initial investment
            locked_capital = 0  # Capital is now in BTC
            state = "BTC"

        elif state == "BTC" and btc > 0:
            if price >= data.iloc[0]["price"] * (1 + sell_thresh):  # Use first price as reference
                usd_received = btc * price * (1 - FEE)
                current_value = btc * price

                if current_value > total_cost_basis:
                    # There is profit - return cost basis to locked_capital, profit to profit_only
                    locked_capital += total_cost_basis
                    profit_amount = current_value - total_cost_basis
                    profit_only += profit_amount * (1 - FEE)  # Apply fee to profit portion
                else:
                    # Loss - all proceeds go back to locked_capital
                    locked_capital += usd_received

                btc = 0
                total_cost_basis = 0
                state = "USD"

            elif price <= data.iloc[0]["price"] * (1 - buy_thresh) and profit_only > 0:
                usd_to_invest = profit_only
                btc_bought = (usd_to_invest * (1 - FEE)) / price
                btc += btc_bought
                total_cost_basis += usd_to_invest
                profit_only = 0
                state = "BTC"

def simulate_strategy_profit_only(data, sell_thresh, buy_thresh, initial_usd, debug=False):
    # Profit-only strategy: extract initial investment, reinvest only profits
    initial_capital = initial_usd  # This will be returned, not reinvested
    available_capital = 0         # Profit available for reinvestment
    btc = 0.0
    total_cost_basis = 0.0       # tracks total amount invested in BTC
    last_buy_price = None        # Track our buy price for threshold calculations
    state = "USD"

    # Minimum thresholds to avoid floating point issues
    MIN_BTC_HOLDINGS = 1e-8  # Minimum BTC to consider holding
    MIN_CAPITAL = 1e-6       # Minimum capital to consider for trading

    if debug:
        print(f"Starting with initial_capital: ${initial_capital}, available_for_reinvestment: ${available_capital}")

    for i, (_, row) in enumerate(data.iterrows()):
        price = row["price"]

        if state == "USD" and available_capital > MIN_CAPITAL:
            # Buy with profit capital
            usd_to_invest = available_capital
            btc_bought = (usd_to_invest * (1 - FEE)) / price
            btc += btc_bought
            total_cost_basis += usd_to_invest
            last_buy_price = price
            if debug:
                print(f"Day {i}: BOUGHT with profit ${usd_to_invest:.2f} -> {btc_bought:.8f} BTC at ${price:.2f}")
            available_capital = 0
            state = "BTC"

        elif state == "USD" and btc < MIN_BTC_HOLDINGS and initial_capital > MIN_CAPITAL:
            # First trade: use initial capital
            usd_to_invest = initial_capital
            btc_bought = (usd_to_invest * (1 - FEE)) / price
            btc += btc_bought
            total_cost_basis += usd_to_invest
            last_buy_price = price
            if debug:
                print(f"Day {i}: FIRST BUY with initial ${usd_to_invest:.2f} -> {btc_bought:.8f} BTC at ${price:.2f}")
            initial_capital = 0  # Initial capital is now invested
            state = "BTC"

        elif state == "BTC" and btc > MIN_BTC_HOLDINGS:
            if price >= last_buy_price * (1 + sell_thresh):
                usd_received = btc * price * (1 - FEE)
                current_value = btc * price

                if debug:
                    print(f"Day {i}: SELL signal - BTC: {btc:.8f}, Price: ${price:.2f}, Buy price: ${last_buy_price:.2f}")

                if usd_received > total_cost_basis:
                    # Profit! Return initial investment equivalent, reinvest profit
                    initial_return = min(total_cost_basis, usd_received)  # Don't return more than received
                    profit_amount = usd_received - initial_return
                    available_capital += profit_amount
                    if debug:
                        print(f"  -> Returned initial: ${initial_return:.2f}, Profit for reinvestment: ${profit_amount:.6f}")
                        print(f"  -> Total available for reinvestment: ${available_capital:.6f}")
                else:
                    # Loss - just add back what we can
                    available_capital += usd_received
                    if debug:
                        print(f"  -> Loss - recovered ${usd_received:.2f} for reinvestment")

                btc = 0.0
                total_cost_basis = 0.0
                last_buy_price = None
                state = "USD"

            elif price <= last_buy_price * (1 - buy_thresh) and available_capital > MIN_CAPITAL:
                usd_to_invest = available_capital
                btc_bought = (usd_to_invest * (1 - FEE)) / price
                btc += btc_bought
                total_cost_basis += usd_to_invest
                last_buy_price = (last_buy_price + price) / 2  # Simple average for now
                if debug:
                    print(f"Day {i}: REBUY with profit ${usd_to_invest:.2f} -> {btc_bought:.8f} BTC at ${price:.2f}")
                available_capital = 0
                state = "BTC"

    # Final value shows ONLY profits made (initial capital is preserved separately)
    final_btc_value = btc * data.iloc[-1]["price"] if btc > MIN_BTC_HOLDINGS else 0
    total_profit = available_capital + final_btc_value

    # Calculate total return including initial capital recovery
    final_total_value = initial_usd + total_profit

    if debug:
        print(f"Final: profit_capital=${available_capital:.6f}, btc_value=${final_btc_value:.6f}")
        print(f"Total profit made: ${total_profit:.6f} (initial ${initial_usd:.0f} preserved separately)")
        print(f"Final total value (initial + profit): ${final_total_value:.2f}")
        profit_percentage = (total_profit / initial_usd) * 100
        print(f"Profit percentage: {profit_percentage:.2f}%")

    return total_profit

# Calculate buy-and-hold benchmark
initial_price = data.iloc[0]["price"]
final_price = data.iloc[-1]["price"]
btc_bought_hold = INITIAL_USD / initial_price
print(f"BTC bought in buy & hold: {btc_bought_hold:.8f}")
hold_final_value = btc_bought_hold * final_price
hold_percentage = ((hold_final_value - INITIAL_USD) / INITIAL_USD) * 100

# Run the profit-only strategy (set debug=True to see detailed output)
INITIAL_INVEST = INITIAL_USD
final_value = simulate_strategy_profit_only(data, MY_SELL_THRESHOLD, MY_BUY_THRESHOLD, INITIAL_INVEST, debug=DEBUG_MODE)

if DEBUG_MODE:
    print(f"\nPure profit made (initial ${INITIAL_INVEST:.0f} preserved): ${final_value:.2f}")
else:
    print(f"\nProfit-only strategy result: ${final_value:.2f}")

# Display comparison
print(f"\n{'='*60}")
print(f"BENCHMARK COMPARISON (Initial Investment: ${INITIAL_USD:,.0f})")
print(f"{'='*60}")
print(f"Buy & Hold Strategy:     ${hold_final_value:,.2f} ({hold_percentage:+.1f}%)")
print(f"Your Strategy:           ${my_row.iloc[0]['final_value']:,.2f} ({my_row.iloc[0]['percent_change']:+.1f}%)")
print(f"Profit-Only Strategy:    ${final_value:.2f}")
print(f"{'='*60}")
print(f"Strategy vs Buy & Hold:  {my_row.iloc[0]['percent_change'] - hold_percentage:+.1f}% better")
print(f"{'='*60}")

# =====================
# Alternative Strategies Reports
# =====================
run_and_report_strategies(data, label="Filtered Window")
run_and_report_strategies(data_full, label="Full History")
parameter_sweep(data_full, label="Full History")
