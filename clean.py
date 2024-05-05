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

# Apply the clean_note_value function to the NOTE column
df['NOTE'] = df['NOTE'].apply(clean_note_value)

# Save the cleaned DataFrame back to a CSV file
df.to_csv(base+"/cleaned_note_result.csv", index=False)

print("Data cleaned and saved as cleaned_output.csv")
