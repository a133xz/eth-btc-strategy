import pandas as pd
import json
from itertools import product

# =====================
# CONFIGURATION
# =====================
FEE = 0.001  # 0.1% trading fee
INITIAL_USD = 7000

PRINT_DETAILED = False  # Set to False to skip printing detailed transactions

# Strategy thresholds (percentage changes)
SELL_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
BUY_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

# Date range filter
START_DATE = "2025-09-08"
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
