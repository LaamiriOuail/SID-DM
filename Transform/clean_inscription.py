import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

import pandas as pd

# Load the Excel file
excel_file = 'DATA/NV_INSRIT_2014_2015_2016(1) (1).xlsx'
df = pd.read_excel(excel_file, skiprows=1)

# Remove null rows and columns
cleaned_df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')




# Rename the "CODE" column to "CODE_ETU"
cleaned_df = cleaned_df.rename(columns={'CODE': 'CODE_ETU'})

# Save the cleaned DataFrame back to Excel
cleaned_excel_file = 'DATA/RES/Inscription.csv'
cleaned_df.to_csv(cleaned_excel_file, index=False)

print("Excel file cleaned and saved successfully.")
