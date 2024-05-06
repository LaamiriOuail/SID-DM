# Project Documentation

This project involves managing data related to students, modules, and their academic performance. Below is the documentation for the tables used in this project and their relationships.

## Setup :

## Install requirements :
```bash
$ pip3 install -r requirements.txt
```

## Run app :
```bash
$ streamlit run ./main.py
```

## Build docker image : 
```bash
$ docker build -t my_streamlit_app . 
```
## Run container:
```bash
$ docker run -p 8501:8501 my_streamlit_app
```
go to http://localhost:8501

## Table Documentation

### Etudiant Table

- **Columns**:
  - `CODE_ETU`: Unique identifier for each student.
  - `PARCOURS`: Academic program or track followed by the student.

### Module Table

- **Columns**:
  - `CODE_MOD`: Unique identifier for each module.
  - `NAME`: Name of the module.
  - `PARCOURS`: Academic program or track to which the module belongs.
  - `SEMESTRE`: Semester in which the module is offered.

### Notes Par Module Table

- **Columns**:
  - `CODE_MOD`: Identifier of the module for which the note is recorded.
  - `NOTE`: The score or grade obtained in the module.
  - `RESULT`: The result of the module (e.g., 'V' for validated, 'NV' for not validated).
  - `SESSION`: Session in which the module was taken.
  - `ANNE`: Academic year in which the module was taken.
  - `CODE_ETU`: Identifier of the student associated with the module.

### Notes Finale Table

- **Columns**:
  - `CODE_ETU`: Identifier of the student.
  - `NOTE_S1`: Sum of the notes for semester 1.
  - `RESULT_S1`: Result for semester 1.
  - `NOTE_S2`: Sum of the notes for semester 2.
  - `RESULT_S2`: Result for semester 2.
  - `NOTE_S3`: Sum of the notes for semester 3.
  - `RESULT_S3`: Result for semester 3.
  - `NOTE_S4`: Sum of the notes for semester 4.
  - `RESULT_S4`: Result for semester 4.
  - `NOTE_ANNE_1`: Sum of the notes for academic year 1.
  - `RESULT_ANNE_1`: Result for academic year 1.
  - `NOTE_ANNE_2`: Sum of the notes for academic year 2.
  - `RESULT_ANNE_2`: Result for academic year 2.
  - `PARCOURS`: Academic program or track of the student.
  - `ANNE_1`: Academic year 1.
  - `ANNE_2`: Academic year 2.

## Relationships

- The `Etudiant` table is related to the `Notes Finale` and `Notes Par Module` tables through the `CODE_ETU` column.
- The `Module` table is related to the `Notes Par Module` table through the `CODE_MOD` column.
- The `Notes Finale` table aggregates information from the `Notes Par Module` table to provide summarized results for each student.
