import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

import pandas as pd

base = "DATA/RES/"


# Read the merged CSV file into a DataFrame
df_result = pd.read_csv(base+"/Notes Finale.csv")


# Function to set the value of the RESULT column based on the NOTE column
def set_result(note):
    if note==0:
        return 'NP'
    elif note >= 10:
        return 'V'
    elif note < 10:
        return 'NV'
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


df_result["NOTE_DEUST"]=round((df_result["NOTE_ANNE_1"]+df_result["NOTE_ANNE_2"])/2,2)

df_result['RESULT_DEUST'] = df_result['NOTE_DEUST'].apply(set_result)

fill_missing_by_privious(df_result,"ANNE_1")
fill_missing_by_privious(df_result,"ANNE_2")

# Save the cleaned DataFrame back to a CSV file
df_result.to_csv(base+"/Notes Finale 1.csv", index=False)

print(f"Data cleaned and saved as {base+"/Notes Finale 1.csv"}")
