import pandas as pd

base = "DATA/RES/"


# Read the merged CSV file into a DataFrame
df = pd.read_csv(base+"/result.csv")

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


# Apply the clean_note_value function to the NOTE column
df['NOTE'] = df['NOTE'].apply(clean_note_value)

# Apply the set_result function to the NOTE column and create the RESULT column
df['RESULT'] = df['NOTE'].apply(set_result)

# Apply the clean_note_value function to the NOTE column
df['NOTE'] = df['NOTE'].apply(clean_null_note_value)

fill_missing_by_privious(df,"ANNE")
fill_missing_by_privious(df,"SESSION")

# Save the cleaned DataFrame back to a CSV file
df.to_csv(base+"/cleaned_module_result.csv", index=False)

print("Data cleaned and saved as cleaned_output.csv")
