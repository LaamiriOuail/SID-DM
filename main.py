import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

base_data_dir:str=str(Path("DATA/RES"))
# Sample data
data = {
    "Etudiant": pd.read_csv(f"{base_data_dir}/Etudiant.csv"),
    "Module": pd.read_csv(f"{base_data_dir}/Module.csv"),
    "Notes Finale": pd.read_csv(f"{base_data_dir}/Notes Finale.csv"),
    "Notes Par Module": pd.read_csv(f"{base_data_dir}/Notes Par Module.csv"),
    "Inscription" : pd.read_csv(f"{base_data_dir}/Inscription.csv")
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


def plot_inscription_evolution_histogramme_grouper(x_col, y_col, hue_col=None):
    """
    Plots the grouped histogram (bar plot) for the evolution of inscription rates.

    Args:
        x_col (str): The column to use for the x-axis.
        y_col (str): The column to use for the y-axis (e.g., "Number of Inscriptions").
        hue_col (str, optional): The column to use for the hue (optional). Defaults to None.
    """

    if hue_col!=None:
        if x_col==y_col and x_col!=hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,hue_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and x_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and y_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and y_col!=hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col,hue_col]).size().reset_index(name="Number of Inscriptions")
        if x_col==y_col and y_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col]).size().reset_index(name="Number of Inscriptions")
    else:
        if x_col==y_col:
            inscription_evolution = data['Inscription'].groupby([x_col]).size().reset_index(name="Number of Inscriptions")
        else:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")

    # Plot grouped histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    if hue_col:
        sns.barplot(data=inscription_evolution, x=x_col, y="Number of Inscriptions", hue=hue_col, ax=ax)
    else:
        sns.barplot(data=inscription_evolution, x=x_col, y="Number of Inscriptions", ax=ax)
        
    ax.set_title(f"Répartition des inscriptions par {y_col} et par {x_col}")
    ax.set_xlabel(f"{x_col} d'obtention du bac")
    ax.set_ylabel("Nombre d'inscriptions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig

# Function to plot evolution of inscription rates by bac series
def plot_inscription_evolution_histogramme_plot(x_col, y_col, hue_col=None):
    """
    Plots the evolution of inscription rates by bac series, optionally adding hue.

    Args:
        x_col (str): The column to use for the x-axis.
        y_col (str): The column to use for the y-axis (e.g., "Number of Inscriptions").
        hue_col (str, optional): The column to use for the hue (optional). Defaults to None.
    """

    if hue_col!=None:
        if x_col==y_col and x_col!=hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,hue_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and x_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and y_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and y_col!=hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col,hue_col]).size().reset_index(name="Number of Inscriptions")
        if x_col==y_col and y_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col]).size().reset_index(name="Number of Inscriptions")
    else:
        if x_col==y_col:
            inscription_evolution = data['Inscription'].groupby([x_col]).size().reset_index(name="Number of Inscriptions")
        else:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")

    # Plot based on selected plot type
    fig, ax = plt.subplots(figsize=(10, 6))

    if hue_col:
        sns.histplot(data=inscription_evolution, x=x_col, y="Number of Inscriptions", hue=hue_col, multiple="stack", ax=ax)
    else:
        sns.histplot(data=inscription_evolution, x=x_col, y="Number of Inscriptions", multiple="stack", ax=ax)

    ax.set_title(f"Répartition des inscriptions par {y_col} et par {x_col}")
    ax.set_xlabel(f"{x_col} d'obtention du bac")
    ax.set_ylabel("Nombre d'inscriptions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig

# Function to plot evolution of inscription rates by bac series line chart
def plot_inscription_evolution_line_plot(x_col, y_col, hue_col=None):
    """
    Plots the evolution of inscription rates by bac series as a line chart, optionally adding hue.

    Args:
        x_col (str): The column to use for the x-axis.
        y_col (str): The column to use for the y-axis (e.g., "Number of Inscriptions").
        hue_col (str, optional): The column to use for the hue (optional). Defaults to None.
    """
    # Group by x_col, y_col and optionally hue_col
    if hue_col!=None:
        if x_col==y_col and x_col!=hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,hue_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and x_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and y_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")
        if x_col!=y_col and y_col!=hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col,hue_col]).size().reset_index(name="Number of Inscriptions")
        if x_col==y_col and y_col==hue_col:
            inscription_evolution = data['Inscription'].groupby([x_col]).size().reset_index(name="Number of Inscriptions")
    else:
        if x_col==y_col:
            inscription_evolution = data['Inscription'].groupby([x_col]).size().reset_index(name="Number of Inscriptions")
        else:
            inscription_evolution = data['Inscription'].groupby([x_col,y_col]).size().reset_index(name="Number of Inscriptions")

    # Plot line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    if hue_col:
        sns.lineplot(data=inscription_evolution, x=x_col, y="Number of Inscriptions", hue=hue_col, ax=ax)
    else:
        sns.lineplot(data=inscription_evolution, x=x_col, y="Number of Inscriptions", ax=ax)

    ax.set_title(f"Evolution des taux d'inscription par {y_col} et par {x_col}")
    ax.set_xlabel(f"{x_col} d'obtention du bac")
    ax.set_ylabel("Nombre d'inscriptions")

    return fig


# Main function
def evolution_by_inscription():
  st.title("Evolution des taux d'inscription par série de bac et par année, ville, etc..")

  # Sidebar and plot type selection
  plot_types = ["Select Chart", "Line chart", "Histogramme empilé","Histogramme groupé"]
  selected_plot_type = st.sidebar.selectbox("Select Plot Type", plot_types)
  # Initialize session state
  if "selected_x_column" not in st.session_state:
    st.session_state.selected_x_column = None

  # Sidebar
  st.sidebar.header("Customize Plot")
  x_column = st.sidebar.selectbox("Select X-axis Column", data["Inscription"].columns)
  y_column = st.sidebar.selectbox("Select Y-axis Column", data["Inscription"].columns)
  hue_col = st.sidebar.selectbox("Select Hue Column (Optional)", [None] + data["Inscription"].columns.tolist())

  # Plot based on selected plot type
  if True:
    if selected_plot_type == "Histogramme empilé":
        fig = plot_inscription_evolution_histogramme_plot(x_column, y_column, hue_col)
    elif selected_plot_type == "Line chart":
        fig = plot_inscription_evolution_line_plot(x_column, y_column, hue_col)
    elif selected_plot_type == "Histogramme groupé":
        fig = plot_inscription_evolution_histogramme_grouper(x_column, y_column, hue_col)
    if selected_plot_type!="Select Chart":
        # Display plot
        st.pyplot(fig)

    




def evolution_by_gender():
    st.title("Evolution des taux d'inscription par sexe, par filière et par année")
    # Add content for this page here

def evolution_by_bac_series():
    st.title("Evolution des taux d'inscription par série de baccalauréat pour chaque filière")
    # Add content for this page here

def success_failure_rate():
    st.title("Taux de réussite, d’échec et d’acquisition par module, session, semestre, année pour la filière MIPC, BCG, GEGM")
    # Add content for this page here

# Sidebar
st.sidebar.header("Navigation")

pages = [
    "Display Table", 
    "Taux d'inscription",
    "Evolution par sexe",
    "Evolution par série de bac pour chaque filière",
    "Taux de réussite, d'échec et d'acquisition"
]
selected_pages = st.sidebar.selectbox("Select Page", pages)

# Get user-selected pages
if "Display Table" in selected_pages:
    selected_table = st.sidebar.selectbox("Select Table", list(data.keys()))
    num_rows = st.sidebar.number_input("Number of Rows to Display", min_value=1, value=5)
    search_column = st.sidebar.selectbox("Search Column (Optional)", [None] + data[selected_table].columns.tolist())
    search_query = st.sidebar.text_input("Search", "")
    display_table_page()

# Display selected page
elif "Taux d'inscription"  in selected_pages:
    evolution_by_inscription()
elif "Evolution par sexe"  in selected_pages:
    evolution_by_gender()
elif "Evolution par série de bac pour chaque filière"  in selected_pages:
    evolution_by_bac_series()
elif "Taux de réussite, d'échec et d'acquisition"  in selected_pages:
    success_failure_rate()