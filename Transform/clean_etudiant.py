import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

import pandas as pd

base = "DATA/RES/"


# Read the merged CSV file into a DataFrame
df_etudiant = pd.read_csv(base+"/Etudiant.csv")


# Drop duplicates based on CODE_ETU and CODE_MOD
df_etudiant.drop_duplicates(subset=['CODE_ETU', 'PARCOURS'], inplace=True)

# Save the cleaned DataFrame back to a CSV file
df_etudiant.to_csv(base+"/Etudiant.csv", index=False)