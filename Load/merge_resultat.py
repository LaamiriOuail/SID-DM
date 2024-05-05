import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

import pandas as pd
import glob

# Define input and output paths
input_folder = "DATA/Module/Resultat"
output_file = "DATA/RES/result.csv"

# Get list of CSV files in the Module folder
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

# Initialize an empty list to store DataFrames
dfs = []

# Read each CSV file into a DataFrame and append to the list
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame as a CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV files saved as {output_file}")
