import os
import csv
import pandas as pd
import openpyxl

def delete_empty_rows(worksheet):
    # Delete first row
    worksheet.delete_rows(1)

    # Loop through rows in reverse order and delete rows that are completely empty
    for row in reversed(range(1, worksheet.max_row + 1)):
        empty_row = True
        for cell in worksheet[row]:
            if cell.value is not None:
                empty_row = False
                break
        if empty_row:
            worksheet.delete_rows(row)

# Function to process Excel file and save as CSV
def process_excel_file(file_path):
    try:
        # Load Excel file
        workbook = openpyxl.load_workbook(file_path)
        worksheet = workbook.active

        # Delete first row and all other empty rows
        delete_empty_rows(worksheet)


        # Extract filename
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Define output directory
        output_dir = os.path.join("DATA", "Module","Resultat")
        os.makedirs(output_dir, exist_ok=True)

        # Save the modified workbook
        modified_file_path = os.path.join(output_dir, file_name + "_modified.xlsx")
        workbook.save(modified_file_path)

        # Read the modified Excel file into a pandas DataFrame
        df = pd.read_excel(modified_file_path)

        # Open a CSV file for writing
        csv_filename = os.path.join(output_dir, file_name + ".csv")
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['CODE_MOD', 'NOTE', 'RESULT', 'SESSION', 'ANNE', 'CODE_ETU'])
            id_ = ""
            note = ""
            result = ""
            session = ""
            anne = ""
            ids:list=[]
            # Iterate through rows and insert into the CSV
            for index, row in df.iterrows():
                if row['COD_ETU']:
                    code_etu = row['COD_ETU']
                for col_name, value in row.items():
                    if col_name.startswith('Note :'):
                        id_ = col_name.replace("Note :", "").strip()
                        note = value
                    if id_ in col_name:
                        if col_name.startswith('Résultat :'):
                            result = value
                        elif col_name.startswith('Session :'):
                            session = value
                        elif col_name.startswith('Année :'):
                            anne = value
                    if id_ and note and result and session and anne:
                        # Write to CSV
                        writer.writerow([id_, note, result, session, anne, code_etu])
                        id_ = ""
                        note = ""
                        result = ""
                        session = ""
                        anne = ""

        print(f"Processed {file_path} and saved CSV as {csv_filename}")
        os.remove(modified_file_path)

    except Exception as e:
        os.remove(modified_file_path)
        print(f"Error processing {file_path}: {e}")

# Function to traverse directories and process Excel files
def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                file_path = os.path.join(root, file)
                process_excel_file(file_path)

# Process the current directory
current_directory = os.getcwd()
process_directory(current_directory)
