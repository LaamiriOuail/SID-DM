import streamlit as st
import pandas as pd
from pathlib import Path

base_data_dir:str=str(Path("DATA/RES"))
# Sample data
data = {
    "Etudiant": pd.read_csv(f"{base_data_dir}/Etudiant.csv"),
    "Module": pd.read_csv(f"{base_data_dir}/Module.csv"),
    "Notes Finale": pd.read_csv(f"{base_data_dir}/Notes Finale.csv"),
    "Notes Par Module": pd.read_csv(f"{base_data_dir}/Notes Par Module.csv"),
}

# Page functions
def display_table_page():
    st.title("Display Data")
    st.header(f"Selected Table: {selected_table}")
    if selected_table:
        df = data[selected_table]
        if search_query and search_column:
            df = df[df[search_column].astype(str).str.contains(search_query)]
        elif search_query:
            df = df[df.apply(lambda row: search_query in str(row), axis=1)]
        st.dataframe(df.head(num_rows))

def other_page():
    st.title("Other Page")
    # Add content for other page here

# Sidebar
st.sidebar.header("Navigation")
pages = ["Display Table", "Other Page"]
selected_pages = st.sidebar.multiselect("Select Page", pages, default=["Display Table"])

# Get user-selected pages
if "Display Table" in selected_pages:
    selected_table = st.sidebar.selectbox("Select Table", list(data.keys()))
    num_rows = st.sidebar.number_input("Number of Rows to Display", min_value=1, value=5)
    search_query = st.sidebar.text_input("Search", "")
    search_column = st.sidebar.selectbox("Search Column (Optional)", [None] + data[selected_table].columns.tolist())
    display_table_page()

if "Other Page" in selected_pages:
    other_page()