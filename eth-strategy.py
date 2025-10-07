import pandas as pd
import json
from itertools import product

# =====================
# CONFIGURATION
# =====================
FEE = 0.001  # 0.1% trading fee
INITIAL_USD = 500

PRINT_DETAILED = False  # Set to False to skip printing detailed transactions

# Strategy thresholds (percentage changes)
SELL_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
BUY_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

# Date range filter
START_DATE = "2020-09-08"
END_DATE = "2025-10-07"

# Your chosen thresholds
MY_SELL_THRESHOLD = 0.07
MY_BUY_THRESHOLD = 0.01

# =====================
# Load BTC price data
# =====================
with open("btc_last5y.json", "r") as f:
    raw_data = json.load(f)

data = pd.DataFrame(raw_data)
data["date"] = pd.to_datetime(data["date"])
data = data.sort_values("date").reset_index(drop=True)

if START_DATE:
    data = data[data["date"] >= pd.to_datetime(START_DATE)]
if END_DATE:
    data = data[data["date"] <= pd.to_datetime(END_DATE)]

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
    btc = 0
    total_cost_basis = 0         # tracks total amount invested in BTC
    last_buy_price = None        # Track our buy price for threshold calculations
    state = "USD"

    if debug:
        print(f"Starting with initial_capital: ${initial_capital}, available_for_reinvestment: ${available_capital}")

    for i, (_, row) in enumerate(data.iterrows()):
        price = row["price"]

        if state == "USD" and available_capital > 0:  # Only invest if we have any capital
            # Buy with profit capital
            usd_to_invest = available_capital
            btc_bought = (usd_to_invest * (1 - FEE)) / price
            btc += btc_bought
            total_cost_basis += usd_to_invest
            last_buy_price = price
            if debug:
                print(f"Day {i}: BOUGHT with profit ${usd_to_invest:.2f} -> {btc_bought:.6f} BTC at ${price:.2f}")
            available_capital = 0
            state = "BTC"

        elif state == "USD" and btc == 0 and initial_capital > 0:
            # First trade: use initial capital
            usd_to_invest = initial_capital
            btc_bought = (usd_to_invest * (1 - FEE)) / price
            btc += btc_bought
            total_cost_basis += usd_to_invest
            last_buy_price = price
            if debug:
                print(f"Day {i}: FIRST BUY with initial ${usd_to_invest:.2f} -> {btc_bought:.6f} BTC at ${price:.2f}")
            initial_capital = 0  # Initial capital is now invested
            state = "BTC"

        elif state == "BTC" and btc > 0:
            if price >= last_buy_price * (1 + sell_thresh):
                usd_received = btc * price * (1 - FEE)
                current_value = btc * price

                if debug:
                    print(f"Day {i}: SELL signal - BTC: {btc:.6f}, Price: ${price:.2f}, Buy price: ${last_buy_price:.2f}")

                if usd_received > total_cost_basis:
                    # Profit! Return initial investment equivalent, reinvest profit
                    initial_return = total_cost_basis
                    profit_amount = usd_received - total_cost_basis
                    available_capital += profit_amount
                    if debug:
                        print(f"  -> Returned initial: ${initial_return:.2f}, Profit for reinvestment: ${profit_amount:.6f}")
                        print(f"  -> Total available for reinvestment: ${available_capital:.6f}")
                else:
                    # Loss - just add back what we can
                    available_capital += usd_received
                    if debug:
                        print(f"  -> Loss - recovered ${usd_received:.2f} for reinvestment")

                btc = 0
                total_cost_basis = 0
                last_buy_price = None
                state = "USD"

            elif price <= last_buy_price * (1 - buy_thresh) and available_capital > 0:
                usd_to_invest = available_capital
                btc_bought = (usd_to_invest * (1 - FEE)) / price
                btc += btc_bought
                total_cost_basis += usd_to_invest
                last_buy_price = (last_buy_price + price) / 2  # Simple average for now
                if debug:
                    print(f"Day {i}: REBUY with profit ${usd_to_invest:.2f} -> {btc_bought:.6f} BTC at ${price:.2f}")
                available_capital = 0
                state = "BTC"

    # Final value shows ONLY profits made (initial capital is preserved separately)
    final_btc_value = btc * data.iloc[-1]["price"] if btc > 0 else 0
    total_profit = available_capital + final_btc_value

    if debug:
        print(f"Final: profit_capital=${available_capital:.6f}, btc_value=${final_btc_value:.6f}")
        print(f"Total profit made: ${total_profit:.6f} (initial ${initial_usd:.0f} preserved separately)")

    return total_profit

# Run the profit-only strategy (set debug=True to see detailed output)
DEBUG_MODE = False  # Set to False for clean output
INITIAL_INVEST = INITIAL_USD
final_value = simulate_strategy_profit_only(data, MY_SELL_THRESHOLD, MY_BUY_THRESHOLD, INITIAL_INVEST, debug=DEBUG_MODE)

if DEBUG_MODE:
    print(f"\nPure profit made (initial ${INITIAL_INVEST:.0f} preserved): ${final_value:.2f}")
else:
    print(f"\nProfit-only strategy result: ${final_value:.2f}")
