import time
import pandas as pd
from dune_client.client import DuneClient

def export_dune_to_csv(api_key, query_id, output_filename, max_retries=5):
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
                wait = 30 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"Failed: {e}")
                return None
            
if __name__ == "__main__":
    # export_dune_to_csv(api_key="v2Ra4kWvZc3D0ilH8A0YBuTFjOPDAi8s", query_id=7958223, output_filename="mint_data.csv")
    # export_dune_to_csv(api_key="v2Ra4kWvZc3D0ilH8A0YBuTFjOPDAi8s", query_id=7958258, output_filename="pool_metadata.csv")
    # export_dune_to_csv(api_key="v2Ra4kWvZc3D0ilH8A0YBuTFjOPDAi8s", query_id=7958265, output_filename="swap_daily_aggregates.csv")
    # export_dune_to_csv(api_key="v2Ra4kWvZc3D0ilH8A0YBuTFjOPDAi8s", query_id=7958349, output_filename="open_events_raw.csv")
    # export_dune_to_csv(api_key="v2Ra4kWvZc3D0ilH8A0YBuTFjOPDAi8s", query_id=7958585, output_filename="open_events.csv")
    # export_dune_to_csv(api_key="akAh5li59kb9jvcARzOHX3tb1rg0xvRC", query_id=7959023, output_filename="closed_events.csv")
    # export_dune_to_csv(api_key="akAh5li59kb9jvcARzOHX3tb1rg0xvRC", query_id=7959074, output_filename="collect.csv")
    # export_dune_to_csv(api_key="VR3zy0lDXLrjXuhUUFOmmESLRWY7I4Xn", query_id=7959074, output_filename=".csv")
    export_dune_to_csv(api_key="IWaJ2xVx3sihjmxOTOQaVCkbjjHCMVaA", query_id=7987495, output_filename="top3blocks__transactions.csv")