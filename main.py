import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree



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
    # Set the title of the web page
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
    st.title("Evolution of enrollment rates by high school series and by year, city, etc.")

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

    



def plot_grouped_histogram(data,x_column, module_name, module_date, module_session:None=None,parcours:None=None):
    # Define color dictionary for consistent coloring
    color_dict = {'V': 'green', 'NV': 'red', 'AC': 'blue', 'NP': 'black'}
    
    # Plot grouped histogram with smaller size
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.histplot(data=data, x=x_column, hue=x_column, multiple="stack", ax=ax, palette=color_dict)
    if module_session:
        ax.set_title(f"Grouped Histogram of Module {module_name} in {module_date}, Session: {module_session}")
    if parcours:
        ax.set_title(f"Grouped Histogram of Module {module_name} in {module_date}, Parcours: {parcours}")
    ax.set_xlabel(f"Result of {module_name}")
    ax.set_ylabel(f"Count of Results of {module_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_pie_chart(data,x_column):
    # Define color dictionary
    color_dict = {'V': 'green', 'NV': 'red', 'AC': 'blue', 'NP': 'black'}
    
    # Plot pie chart with smaller size
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the figsize as per your preference
    counts = data[x_column].value_counts()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=[color_dict[label] for label in counts.index])
    ax.set_title("Pie Chart of Result Distribution")
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return fig


def evolution_by_module():

    st.title("Success rate, failure rate, and acquisition rate by module.")
    
    # Sidebar
    st.sidebar.header("Customize Data")
    selected_year = st.sidebar.multiselect("Select Year (Anne)", sorted(data["Notes Par Module"]["ANNE"].unique()))
    
    # Get unique PARCOURS IDs from the Module dataframe
    parcours_ids = data["Module"]["PARCOURS"].unique()
    selected_parcours = st.sidebar.multiselect("Select Program (Parcours)", sorted(parcours_ids))
    
    # Get unique semesters based on the selected parcours
    semesters = data["Module"][data["Module"]["PARCOURS"].isin(selected_parcours)]["SEMESTRE"].unique()
    selected_semester = st.sidebar.multiselect("Select Semester", sorted(semesters))
    
    # Get unique module names based on the selected parcours and semester
    module_names = data["Module"][(data["Module"]["PARCOURS"].isin(selected_parcours)) & 
                                  (data["Module"]["SEMESTRE"].isin(selected_semester))]["NAME"].unique()
    selected_module = st.sidebar.multiselect("Select Module Name", sorted(module_names))

    # Allow the session to be None (i.e., no filtering by session)
    sessions = sorted(data["Notes Par Module"]["SESSION"].unique())
    selected_session = st.sidebar.multiselect("Select Session", sessions)
    
    # Plot type selection
    plot_types = ["Data","Grouped Histogram", "Pie Chart"]
    selected_plot_types = st.sidebar.multiselect("Select Plot Types or Data", plot_types,default=["Data"])

    

    # Filter the data based on user selections
    if not selected_session:
        # If no session is selected, include all sessions
        filtered_data = data["Notes Par Module"][
            (data["Notes Par Module"]["ANNE"].isin(selected_year)) &
            (data["Notes Par Module"]["CODE_MOD"].isin(data["Module"][(data["Module"]["PARCOURS"].isin(selected_parcours)) & 
                                                                    (data["Module"]["SEMESTRE"].isin(selected_semester)) & 
                                                                    (data["Module"]["NAME"].isin(selected_module))]["CODE_MOD"]))
        ]
    else:
        filtered_data = data["Notes Par Module"][
            (data["Notes Par Module"]["ANNE"].isin(selected_year)) &
            (data["Notes Par Module"]["SESSION"].isin(selected_session)) &
            (data["Notes Par Module"]["CODE_MOD"].isin(data["Module"][(data["Module"]["PARCOURS"].isin(selected_parcours)) & 
                                                                    (data["Module"]["SEMESTRE"].isin(selected_semester)) & 
                                                                    (data["Module"]["NAME"].isin(selected_module))]["CODE_MOD"]))
        ]


    # Plot the selected plot types
    for plot_type in selected_plot_types:
        if plot_type == "Data":
             # Display the filtered data
            st.markdown(f"### Data Table :")
            st.write(filtered_data)
        elif plot_type == "Grouped Histogram":
            # Plot grouped histogram
            st.markdown(f"### Grouped Histogram of Module {selected_module} in {selected_year}, session: {selected_session if selected_session else ['1,2']}:")
            st.pyplot(plot_grouped_histogram(filtered_data,"RESULT",selected_module,selected_year,module_session=selected_session if selected_session else "1,2"))
        elif plot_type == "Pie Chart":
            # Plot pie chart
            st.markdown(f"### Pie Chart of Result Distribution:")
            st.pyplot(plot_pie_chart(filtered_data,"RESULT"))
   

def evolution_by_finale_notes():

    st.title("Success rate, failure rate, and acquisition rate by final grades.")

    # Get existing years without null values
    existing_years = data["Notes Finale"]["ANNE_1"].dropna().unique().tolist() + data["Notes Finale"]["ANNE_2"].dropna().unique().tolist()
    existing_years = sorted(list(set(existing_years)))

    # Sidebar
    st.sidebar.header("Customize Data")
    selected_years = st.sidebar.multiselect("Select Year(s)", existing_years)
    
    selected_year_session = st.sidebar.selectbox("Select Year Session (1,2)", [1,2])

    # Based on the selected year, determine the possible semesters
    possible_semesters:list=[]
    if selected_year_session == 1:
        possible_semesters += [None,1, 2]
    else:
        possible_semesters = [None,3, 4]
    
    selected_semester = st.sidebar.selectbox("Select Semester", possible_semesters)
    
    # Get unique PARCOURS IDs from the Notes Finale dataframe
    parcours_ids = data["Notes Finale"]["PARCOURS"].unique()
    selected_parcours = st.sidebar.multiselect("Select Program (Parcours)", sorted(parcours_ids))
    
    # Plot type selection
    plot_types = ["Data", "Grouped Histogram", "Pie Chart"]
    selected_plot_types = st.sidebar.multiselect("Select Plot Types or Data", plot_types, default=["Data"])


    # Filter the data based on user selections
    filtered_data=None
    result_=""
    if selected_semester and selected_parcours:
        filtered_data = data["Notes Finale"][
            (data["Notes Finale"][f"ANNE_{selected_year_session}"].isin(selected_years)) &
            (data["Notes Finale"][f"RESULT_S{selected_semester}"].notna()) &
            (data["Notes Finale"]["PARCOURS"].isin(selected_parcours))
        ][["CODE_ETU", f"RESULT_S{selected_semester}" , f"NOTE_S{selected_semester}" , "PARCOURS"]]
        result_=f"RESULT_S{selected_semester}"
    elif selected_parcours:
        filtered_data = data["Notes Finale"][
            (data["Notes Finale"][f"ANNE_{selected_year_session}"].isin(selected_years)) &
            (data["Notes Finale"]["PARCOURS"].isin(selected_parcours))
        ][["CODE_ETU", f"RESULT_ANNE_{selected_year_session}" , f"NOTE_ANNE_{selected_year_session}", "PARCOURS"]]
        result_=f"RESULT_ANNE_{selected_year_session}"

    # Plot the selected plot types
    if not (filtered_data is None):
        for plot_type in selected_plot_types:
            if plot_type == "Data":
                # Display the filtered data
                st.markdown("### Data Table:")
                st.write(filtered_data)
            elif plot_type == "Grouped Histogram":
                # Plot grouped histogram
                if selected_years and selected_parcours:
                    st.pyplot(plot_grouped_histogram(filtered_data,result_, "Final Notes", selected_years, parcours=selected_parcours))
                else:
                    st.warning("Enter valid years and parcours")
            elif plot_type == "Pie Chart":
                # Plot pie chart
                if selected_years and selected_parcours:
                    st.pyplot(plot_pie_chart(filtered_data,result_))
                else:
                    st.warning("Enter valid years and parcours")


def evolution_by_diplome():

    st.title("Success rate, failure rate, and acquisition rate by diploma.")
    
    # Sidebar
    st.sidebar.header("Customize Data")
    selected_year = st.sidebar.multiselect("Select Year (Anne)", sorted(list(set(data["Notes Finale"]["ANNE_1"].dropna().unique().tolist() + data["Notes Finale"]["ANNE_2"].dropna().unique().tolist()))))
    
    # Get unique PARCOURS IDs from the Module dataframe
    parcours_ids = data["Module"]["PARCOURS"].unique()
    selected_parcours = st.sidebar.multiselect("Select Program (Parcours)", sorted(parcours_ids))
    
    # Plot type selection
    plot_types = ["Data","Grouped Histogram", "Pie Chart"]
    selected_plot_types = st.sidebar.multiselect("Select Plot Types or Data", plot_types,default=["Data"])

    

    
    # If no session is selected, include all sessions
    filtered_data = data["Notes Finale"][
            (data["Notes Finale"]["ANNE_1"].isin(selected_year))  &
            (data["Notes Finale"]["PARCOURS"].isin(selected_parcours))  
        ]
   


    # Plot the selected plot types
    for plot_type in selected_plot_types:
        if plot_type == "Data":
             # Display the filtered data
            st.markdown(f"### Data Table :")
            st.write(filtered_data)
        elif plot_type == "Grouped Histogram":
            # Plot grouped histogram
            st.markdown(f"### Grouped Histogram in {selected_year}, parcours: {selected_parcours}:")
            if selected_year and selected_parcours:
                st.pyplot(plot_grouped_histogram(filtered_data,"RESULT_DEUST","",selected_year,parcours=selected_parcours))
            else:
                st.warning("Enter valid years and parcours")
        elif plot_type == "Pie Chart":
            # Plot pie chart
            st.markdown(f"### Pie Chart of Result Distribution:")
            if selected_year and selected_parcours:
                st.pyplot(plot_pie_chart(filtered_data,"RESULT_DEUST"))
            else:
                st.warning("Enter valid years and parcours")
   
def clusturing():
    st.title("Clustering")
    
    selected_table = st.sidebar.selectbox("Select Table", list(data.keys()))
    selected_data = data[selected_table]  # Get the selected table data
    selected_rows = st.sidebar.multiselect("Select Rows", list(selected_data.columns), default=list(selected_data.columns))
    
    # Filter selected data with selected rows
    selected_data = selected_data[selected_rows].copy()

    # Handle missing values
    missing_values = selected_data.isnull().sum()
    if missing_values.any():
        st.warning("Warning: Missing values detected! Imputing missing values.")
        for col in selected_data.columns:
            if selected_data[col].dtype == 'object':  # For categorical columns
                imputer = SimpleImputer(strategy="most_frequent")
                selected_data[col] = imputer.fit_transform(selected_data[[col]]).ravel()
            else:  # For numeric columns
                imputer = SimpleImputer(strategy="mean")
                selected_data[col] = imputer.fit_transform(selected_data[[col]]).ravel()

    # Identify and encode categorical columns
    categorical_cols = selected_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        selected_data[col] = encoder.fit_transform(selected_data[col])

    selected_algorithm = st.sidebar.selectbox("Select Algorithm", ["KMeans", "Hierarchical", "DBSCAN"])
    if selected_algorithm != "DBSCAN":
        number_of_clusters = st.sidebar.number_input("Number of Clusters", min_value=2, value=2,max_value=7)

    # Apply selected algorithm
    if selected_algorithm == "KMeans":
        clustering_model = KMeans(n_clusters=number_of_clusters, random_state=42)
    elif selected_algorithm == "Hierarchical":
        clustering_model = AgglomerativeClustering(n_clusters=number_of_clusters)
    elif selected_algorithm == "DBSCAN":
        clustering_model = DBSCAN(eps=0.5, min_samples=20)

    # Fit clustering model and get cluster labels
    cluster_labels = clustering_model.fit_predict(selected_data[selected_rows])

    # Add cluster labels to the data
    selected_data['Cluster'] = cluster_labels

    # Plot type selection
    plot_types = ["Data","PCA Visualization", "t-SNE Visualization","LDA Visualization"]
    selected_plot_types = st.sidebar.multiselect("Select Plot Types or Data", plot_types,default=["Data"])


    # Plot the selected plot types
    for plot_type in selected_plot_types:
        if plot_type == "Data":
             # Display the filtered data
            st.markdown(f"### Data Table :")
            st.write(selected_data)
        elif plot_type == "PCA Visualization":
            # Perform PCA for visualization
            st.markdown(f"### PCA Visualization :")

            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(selected_data[selected_rows])

            # Create a scatter plot for PCA visualization
            plt.figure(figsize=(8, 6))
            for cluster in selected_data['Cluster'].unique():
                plt.scatter(pca_result[selected_data['Cluster'] == cluster, 0],
                            pca_result[selected_data['Cluster'] == cluster, 1],
                            label=f'Cluster {cluster}')
            plt.title('PCA Visualization')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            st.pyplot(plt)

        elif plot_type == "t-SNE Visualization":
            # Perform t-SNE for visualization
            st.markdown(f"### t-SNE Visualization :")

            tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
            tsne_result = tsne.fit_transform(selected_data[selected_rows])

            # Create a scatter plot for t-SNE visualization
            plt.figure(figsize=(8, 6))
            for cluster in selected_data['Cluster'].unique():
                plt.scatter(tsne_result[selected_data['Cluster'] == cluster, 0],
                            tsne_result[selected_data['Cluster'] == cluster, 1],
                            label=f'Cluster {cluster}')
            plt.title('t-SNE Visualization')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            st.pyplot(plt)

        elif plot_type == "LDA Visualization":
            # Perform LDA for visualization
            st.markdown(f"### LDA Visualization :")
            if not (2 > min(len(selected_data[selected_rows].columns),len(selected_data['Cluster'].unique())-1)):
                lda = LDA(n_components=2)  # Ensure you specify 2 components
                lda_result = lda.fit_transform(selected_data[selected_rows], selected_data['Cluster'])

                # Create a scatter plot for LDA visualization
                plt.figure(figsize=(8, 6))
                for cluster in selected_data['Cluster'].unique():
                    plt.scatter(lda_result[selected_data['Cluster'] == cluster, 0],
                                lda_result[selected_data['Cluster'] == cluster, 1],
                                label=f'Cluster {cluster}')
                plt.title('LDA Visualization')
                plt.xlabel('Linear Discriminant 1')
                plt.ylabel('Linear Discriminant 2')
                plt.legend()
                st.pyplot(plt)
            else:
                st.warning(f"Insufficient features or classes for LDA visualization.( n_components=2 cannot be larger than min(n_features={len(selected_data[selected_rows].columns)}, n_classes={len(selected_data['Cluster'].unique())} - 1) ) ")


def decesion_tree_page():
    st.title("Decesion tree")

    selected_table = st.sidebar.selectbox("Select Table", ["Notes Finale","Notes Par Module"])
    selected_data = data[selected_table]  # Get the selected table data
    
    criterion = st.sidebar.selectbox("Select Criterion", ["gini", "entropy", "log_loss"])
    # Select only numerical features for the multiselect
    numerical_features = selected_data.select_dtypes(include=[np.number]).columns
    features = st.sidebar.multiselect("Select Numerical Features", list(numerical_features), default=list(numerical_features))

    # Select the target variable (ensure it's categorical)
    target_candidates = selected_data.columns[selected_data.dtypes == 'object']
    target = st.sidebar.selectbox("Select Target", target_candidates)

    # Remove the target from the list of features
    if target in features:
        features.remove(target)

    # Handle missing values

    X = selected_data[features]
    y = selected_data[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree classifier
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)

    # Plot the decision tree
    fig=plt.figure(figsize=(25,20))
    tree.plot_tree(clf, 
                   feature_names=features,  
                   class_names=clf.classes_,  # Use clf.classes_ to get class names
                   filled=True,
                   rounded=True,
                   proportion=True
                   )
    plt.title("Decision Tree")

    # Display the plot in Streamlit
    st.pyplot(fig)





# Sidebar
st.sidebar.header("Navigation")

pages = [
    "Display Table", 
    "Taux d'inscription",
    "Evolution par Module",
    "Evolution par Semestre,Anne",
    "Evolution par Diplome",
    "Clustering",
    "Decision Tree",
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
elif "Evolution par Module"  in selected_pages:
    evolution_by_module()
elif "Evolution par Semestre,Anne"  in selected_pages:
    evolution_by_finale_notes()
elif "Evolution par Diplome"  in selected_pages:
    evolution_by_diplome()
elif "Clustering" in selected_pages:
    clusturing()
elif "Decision Tree" in selected_pages:
    decesion_tree_page()