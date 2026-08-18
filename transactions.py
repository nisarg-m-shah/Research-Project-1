import requests
import pandas as pd
import time
from datetime import datetime, timedelta

API_KEY = "CG-Yhy55uwtYwcmxy6AMHJwGjwQ"  # paste your key

headers = {"x-cg-demo-api-key": API_KEY}

token_map = {
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "usd-coin",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "weth",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": "wrapped-bitcoin",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "dai",
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "tether",
}

start_date = datetime(2021, 5, 1)
end_date = datetime(2024, 1, 1)
all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

all_prices = []

for token_address, coin_id in token_map.items():
    print(f"Fetching {coin_id} ({len(all_dates)} days)...")
    success_count = 0
    for d in all_dates:
        date_str = d.strftime("%d-%m-%Y")  # CoinGecko format: dd-mm-yyyy
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/history"
        try:
            resp = requests.get(url, params={"date": date_str}, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            price = data.get("market_data", {}).get("current_price", {}).get("usd")
            if price is not None:
                all_prices.append({
                    "token_address": token_address,
                    "symbol": coin_id,
                    "date": d.date(),
                    "price_usd": price
                })
                success_count += 1
        except Exception as e:
            pass  # skip failed days silently, print summary at end
        time.sleep(0.65)  # ~90 calls/min, safely under 100/min limit

    print(f"  Got {success_count}/{len(all_dates)} days")

df = pd.DataFrame(all_prices)
df.to_csv("raw_prices_daily.csv", index=False)
print(f"\nSaved {len(df)} total rows to raw_prices_daily.csv")