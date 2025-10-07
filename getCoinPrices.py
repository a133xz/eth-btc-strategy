import yfinance as yf
import pandas as pd
import datetime
import json

def fetch_btc_history(start_date: str, end_date: str):
    df = yf.download(
        "ETH-USD",
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="column",
        progress=False,
        auto_adjust=False  # avoids FutureWarning
    )

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[-1] for col in df.columns]

    # If Close column missing, try renaming
    if "Close" not in df.columns:
        if "ETH-USD" in df.columns:
            df = df.rename(columns={"ETH-USD": "Close"})
        else:
            raise ValueError(f"Unexpected columns: {df.columns}")

    df = df.dropna(subset=["Close"])

    # Reset index to get a Date column
    df = df.reset_index()

    result = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row['Date']).date()

        # If row['Close'] is a Series (multi-column), take first value
        price = row['Close']
        if isinstance(price, pd.Series):
            price = price.iloc[0]

        result.append({"date": dt.isoformat(), "price": float(price)})

    return result


if __name__ == "__main__":
    today = datetime.date.today()
    five_years_ago = today - datetime.timedelta(days=5 * 365)
    start = five_years_ago.isoformat()
    end = today.isoformat()

    data = fetch_btc_history(start, end)

    with open("eth_last5y.json", "w") as f:
        json.dump(data, f, indent=2)

    print(json.dumps(data[:5], indent=2))
