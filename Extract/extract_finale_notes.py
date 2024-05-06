import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

import pandas as pd

base = "DATA/RES/"
# Step 1: Read etudiant.csv to get student codes and their parcours
etudiant_df = pd.read_csv(base+"/etudiant.csv")

# Step 2: Filter module.csv based on parcours
module_df = pd.read_csv(base+"/module.csv")

# Step 3: Read result.csv to get data about modules associated with students
result_df = pd.read_csv(base+"/cleaned_module_result.csv")



# Step 4: For each student, sum the notes for each semester
# Step 5: Create lists for each semester and populate them with the summed notes
note_s1 = []
note_s2 = []
note_s3 = []
note_s4 = []
anne_1 = []
anne_2 = []

for index, row in etudiant_df.iterrows():
    code_etu = row['CODE_ETU']
    parcours = row['PARCOURS']
    
    # Filter result_df based on code_etu
    student_modules_df = result_df[result_df['CODE_ETU'] == code_etu]
    
    # Filter student_modules_df based on parcours using merge with module_df
    student_modules_df = pd.merge(student_modules_df, etudiant_df, how='inner', on='CODE_ETU')

    # Filter student_modules_df based on parcours using merge with module_df
    student_modules_df = pd.merge(student_modules_df, module_df, how='inner', on=['CODE_MOD','PARCOURS'])
    # Drop duplicates based on CODE_ETU and CODE_MOD
    student_modules_df.drop_duplicates(subset=['CODE_ETU', 'CODE_MOD'], inplace=True)

    # Sum the notes for each semester
    note_s1.append(student_modules_df[student_modules_df['SEMESTRE'] == 1]['NOTE'].sum())
    note_s2.append(student_modules_df[student_modules_df['SEMESTRE'] == 2]['NOTE'].sum())
    note_s3.append(student_modules_df[student_modules_df['SEMESTRE'] == 3]['NOTE'].sum())
    note_s4.append(student_modules_df[student_modules_df['SEMESTRE'] == 4]['NOTE'].sum())
     # Get the first non-null 'ANNE' value for each semester
    anne_1.append(student_modules_df[student_modules_df['SEMESTRE'] == 1]['ANNE'].dropna().iloc[0] if not student_modules_df[student_modules_df['SEMESTRE'] == 1]['ANNE'].dropna().empty else None)
    anne_2.append(student_modules_df[student_modules_df['SEMESTRE'] == 3]['ANNE'].dropna().iloc[0] if not student_modules_df[student_modules_df['SEMESTRE'] == 3]['ANNE'].dropna().empty else None)

def get_result(note):
    if note==0:
        return "NP"
    elif note>=10:
        return "V"
    elif note>=7:
        return "AC"
    elif note<7:
        return "NV"
    

# Step 6: Create a DataFrame containing the required columns
data = {
    'CODE_ETU': etudiant_df['CODE_ETU'],
    'NOTE_S1': [round(note/6,2) for note in note_s1],
    'RESULT_S1' : [get_result(round(note/6,2)) for note in note_s1],
    'NOTE_S2': [round(note/6,2) for note in note_s2],
    'RESULT_S2' : [get_result(round(note/6,2)) for note in note_s2],
    'NOTE_S3': [round(note/6,2) for note in note_s3],
    'RESULT_S3' : [get_result(round(note/6,2)) for note in note_s3],
    'NOTE_S4': [round(note/6,2) for note in note_s4],
    'RESULT_S4' : [get_result(round(note/6,2)) for note in note_s4],
    'NOTE_ANNE_1': [round((note1+note2)/12,2) for note1,note2 in zip(note_s1,note_s2)],
    'RESULT_ANNE_1' : [get_result(round((note1+note2)/12,2)) for note1,note2 in zip(note_s1,note_s2)],
    'NOTE_ANEE_2': [round((note1+note2)/12,2) for note1,note2 in zip(note_s3,note_s4)],
    'RESULT_ANNE_2' : [get_result(round((note1+note2)/12,2)) for note1,note2 in zip(note_s3,note_s4)],
    'PARCOURS': etudiant_df['PARCOURS'],
    'ANNE_1': anne_1,
    'ANNE_2': anne_2
}
final_df = pd.DataFrame(data)

# Step 7: Save the DataFrame to a CSV file
final_df.to_csv(base+"/Notes Finale.csv", index=False)

print("Data saved to notes_by_semestre.csv")




