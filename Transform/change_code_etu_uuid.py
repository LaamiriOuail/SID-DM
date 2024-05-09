import pandas as pd
import uuid

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('DATA/RES/Etudiant.csv')

# Function to generate UUID for each replacement
def generate_uuid():
    return str(uuid.uuid4())

# Replace 'CODE_ETU' column values with UUIDs
df['CODE_ETU'] = df['CODE_ETU'].apply(lambda x: generate_uuid())

# Save the modified DataFrame to a new CSV file
df.to_csv('DATA/RES/Etudiant 3.csv', index=False)

print("Replacement complete. Output saved to output.csv")
