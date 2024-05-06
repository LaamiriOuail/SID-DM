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
        

df_result["NOTE_DEUST"]=round((df_result["NOTE_ANNE_1"]+df_result["NOTE_ANNE_2"])/2,2)

df_result['RESULT_DEUST'] = df_result['NOTE_DEUST'].apply(set_result)


# Save the cleaned DataFrame back to a CSV file
df_result.to_csv(base+"/Notes Finale 1.csv", index=False)

print(f"Data cleaned and saved as {base+"/Notes Finale 1.csv"}")
