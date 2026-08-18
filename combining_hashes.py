import pandas as pd

open_df = pd.read_csv("open_events_raw.csv")
open_delta_df = pd.read_csv("open_events_topup.csv")
close_df = pd.read_csv("close_events.csv")

all_hashes = pd.concat([
    open_df["evt_tx_hash"],
    open_delta_df["evt_tx_hash"],
    close_df["final_evt_tx_hash"]
]).drop_duplicates()

all_hashes.to_frame(name="hash").to_csv("tx_hashes_for_bigquery.csv", index=False)
print(f"{len(all_hashes)} unique transaction hashes")