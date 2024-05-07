import os
import sys
import pandas as pd

base = "DATA/RES/"

# Read the merged CSV file into a DataFrame
df_result = pd.read_csv(base + "/Notes Finale.csv")
df_inscription = pd.read_csv(base + "/Inscription.csv")

# Create a dictionary from df_inscription to map "CODE_ETU" to "SEXE"
code_etu_sex_mapping = dict(zip(df_inscription["CODE_ETU"], df_inscription["SEXE"]))

# Iterate over each row in df_result
for index, row in df_result.iterrows():
    code_etu = row["CODE_ETU"]
    if code_etu in code_etu_sex_mapping:
        # If "CODE_ETU" exists in df_inscription, update "SEXE" column
        df_result.at[index, "SEXE"] = code_etu_sex_mapping[code_etu]
    else:
        # If "CODE_ETU" doesn't exist in df_inscription, set "SEXE" to None
        df_result.at[index, "SEXE"] = None

# Save the updated DataFrame back to a CSV file
df_result.to_csv(base + "/Notes Finale 1.csv", index=False)

print(f"Data cleaned and saved as {base + '/Notes Finale 1.csv'}")
