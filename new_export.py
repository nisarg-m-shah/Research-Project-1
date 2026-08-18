import time
import pandas as pd
from dune_client.client import DuneClient

def export_dune_to_csv(api_key, query_id, output_filename, max_retries=4):
    dune = DuneClient(api_key)
    for attempt in range(max_retries):
        try:
            print(f"Fetching query {query_id} (attempt {attempt + 1})...")
            df = dune.get_latest_result_dataframe(query_id)
            df.to_csv(output_filename, index=False)
            print(f"Saved {len(df)} rows to '{output_filename}'")
            return df
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            elif "402" in str(e):
                print("Quota exhausted on this account — stop retrying, switch accounts.")
                return None
            else:
                print(f"Failed: {e}")
                return None

export_dune_to_csv(api_key="D2vzZygKSdJeq6fG0PNc8DUFU0TouLBe", query_id=8147923, output_filename="raw_prices_dai_usdt.csv")