# /// script
# dependencies = [
#   "requests",
# ]
# ///

"""
Author: oucailab 
Date: 2023-12-13 15:10:31
Description: Download OSI-450-a1 Sea Ice Concentration data.
"""
import os
import argparse
from datetime import datetime, timedelta
import requests

def download_data(start_date, end_date, output_directory):
    # Base URL for OSI-450-a1 (Global Sea Ice Concentration Climate Data Record v3.0)
    base_url = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/conc_450a1_files"

    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        file_date = current_date.strftime("%Y%m%d")
        
        # File version v3p1 as confirmed by user
        file_name = f"ice_conc_nh_ease2-250_cdr-v3p1_{file_date}1200.nc"
        file_url = f"{base_url}/{year}/{month:02d}/{file_name}"
        
        # Target directory structure: output_directory/year/month/
        target_dir = os.path.join(output_directory, str(year), f"{month:02d}")
        os.makedirs(target_dir, exist_ok=True)
        
        output_file = os.path.join(target_dir, file_name)

        if os.path.exists(output_file):
            print(f"Skipping (already exists): {output_file}")
            current_date += timedelta(days=1)
            continue

        try:
            print(f"Downloading: {file_url}")
            # stream=True and chunked reading is safer for large files to ensure integrity
            with requests.get(file_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(output_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Successfully downloaded: {output_file}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_url}: {e}")

        # Move to the next day
        current_date += timedelta(days=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OSI-450-a1 Sea Ice Concentration data.")
    parser.add_argument("-sd", "--start_date", type=str, default="20180809", help="Start date (YYYYMMDD)")
    parser.add_argument("-ed", "--end_date", type=str, default="20180809", help="End date (YYYYMMDD)")
    parser.add_argument("-o", "--output", type=str, default=".", help="Root output directory")

    args = parser.parse_args()

    try:
        sd = datetime.strptime(args.start_date, "%Y%m%d")
        ed = datetime.strptime(args.end_date, "%Y%m%d")
        download_data(sd, ed, args.output)
    except ValueError as e:
        print(f"Date format error: {e}")
