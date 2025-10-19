import os
import requests
import pandas as pd
import click

# ---------------- CONFIG ---------------- #
START_YEAR, END_YEAR = 2025, 2025
OUT_DIR = "master_files"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)
HEADERS = {"User-Agent": "MyAppName contact@myemail.com"}
# ---------------------------------------- #

# -------- Step 1: Get all 10-K/10-Q entries from master.idx -------- #
def fetch_master_index(year, quarter, out):
    """Download one quarterly index and return DataFrame of that CIK."""
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        print(f"Failed {url}")
        return pd.DataFrame()

    with open(f"{out}/master_idx_{year}_{quarter:02d}.txt", encoding="utf-8", mode="w") as f:
        f.write(r.text)

@click.command()
@click.option('--start', default = START_YEAR, help = "download data since which year?")
@click.option('--end', default = END_YEAR, help = "download data til which year?")
@click.option('--out', default = OUT_DIR, help = "the folder path to save the files")
def run(start, end, out):
    for year in range(start, end):
        for quarter in range(1, 5):
            print(f"Fetching {year} Q{quarter}...")
            fetch_master_index(year, quarter, out) 


if __name__ == '__main__':
    run()
