import shutil
import os
import re

current_directory = "."

# Define a regular expression pattern for matching filenames
pattern = r"^ice_conc_nh_ease2-250_cdr-v3p1_(\d{4})(\d{2})\d{6}\.nc$"

for filename in os.listdir(current_directory):
    match = re.match(pattern, filename)

    if os.path.isfile(filename) and match:
        year, month = match.groups()
        year_directory = os.path.join(current_directory, year)
        month_directory = os.path.join(year_directory, month)

        os.makedirs(month_directory, exist_ok=True)

        source_path = os.path.join(current_directory, filename)
        target_path = os.path.join(month_directory, filename)

        shutil.move(source_path, target_path)
