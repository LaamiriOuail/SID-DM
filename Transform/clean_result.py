import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

import pandas as pd

base = "DATA/RES/"


# Read the merged CSV file into a DataFrame
df_result = pd.read_csv(base+"/result.csv")
df_module = pd.read_csv(base+"/module.csv")

# Function to clean the note values
def clean_note_value(note):
    if note > 20:
        powers_of_10 = [10, 100, 1000, 10000, 100000]
        for power in powers_of_10:
            if note / power <= 20:
                return note / power
    return note

# Function to set the value of the RESULT column based on the NOTE column
def set_result(note):
    if note >= 10:
        return 'V'
    elif note < 10 and note >= 7:
        return 'AC'
    elif note < 7:
        return 'NV'
    else:
        return 'NP'

# Function to clean the note values
def clean_null_note_value(note):
    if pd.isnull(note):
        return 0  # Change null values to 0
    return note

# Function to fill missing years in the ANNE column with the value from the previous row
def fill_missing_by_privious(df,column):
    if column in df.columns:
        previous_year = None
        for index, row in df.iterrows():
            if pd.isnull(row[column]):
                df.at[index, column] = previous_year
            else:
                previous_year = row[column]
    else:
        print(f'{column} not a column in dataframe')

# Filter the DataFrame to remove rows where CODE_ETU is null
df_result.dropna(subset=['CODE_ETU'],inplace=True)

# Apply the clean_note_value function to the NOTE column
df_result['NOTE'] = df_result['NOTE'].apply(clean_note_value)

# Apply the set_result function to the NOTE column and create the RESULT column
df_result['RESULT'] = df_result['NOTE'].apply(set_result)

# Apply the clean_note_value function to the NOTE column
df_result['NOTE'] = df_result['NOTE'].apply(clean_null_note_value)

fill_missing_by_privious(df_result,"ANNE")
fill_missing_by_privious(df_result,"SESSION")

# Drop duplicates based on CODE_ETU and CODE_MOD
df_result.drop_duplicates(subset=['CODE_ETU', 'CODE_MOD'], inplace=True)

# Extract unique values from the 'ID' column of df_module
valid_ids = df_module['ID'].unique()
# Filter df to keep only rows where 'CODE_MOD' is in the valid_ids list
df_result = df_result[df_result['CODE_MOD'].isin(valid_ids)]

# Save the cleaned DataFrame back to a CSV file
df_result.to_csv(base+"/Notes Par Module.csv", index=False)

print("Data cleaned and saved as cleaned_output.csv")
