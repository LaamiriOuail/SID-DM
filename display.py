import pandas as pd 

df_result = pd.read_csv("DATA/RES/cleaned_module_result.csv")


# Get the count of null values for each column
null_counts = df_result.isnull().sum()

print("Null value counts for each column:")
print(null_counts)

# Group the DataFrame by 'CODE_ETU' and count the number of rows in each group
code_etu_counts = df_result.groupby('CODE_ETU').size()

# Iterate over each code_etu and its corresponding count
for code_etu, count in code_etu_counts.items():
    print(f"CODE_ETU: {code_etu}, Number of Rows: {count}")

