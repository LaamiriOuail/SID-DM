import pandas as pd 

df = pd.read_csv("DATA/RES/cleaned_module_result.csv")


# Get the count of null values for each column
null_counts = df.isnull().sum()

print("Null value counts for each column:")
print(null_counts)

