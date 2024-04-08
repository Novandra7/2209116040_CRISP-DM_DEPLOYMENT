import streamlit as st
import pandas as pd
from streamlit_option_menu import *
from function import *

df = pd.read_csv('Data Cleaned.csv')

with st.sidebar :
    selected = option_menu('Iris Flower',['Introducing','Data Distribution','Relation','Composition & Comparison','Predict','Clustering'],default_index=0)

if (selected == 'Introducing'):
    st.title('Types of Iris Flower')
    st.write("""
    Explore the **Iris Flower** dataset and build a classifier to predict the species.
    """)
    st.image('img/variant.png', caption='Iris Flower', use_column_width=True)
    st.title("Iris Flower Information")
    st.write("Select a type of iris flower to learn more about it:")

    iris_types = ["Setosa", "Versicolor", "Virginica"]
    iris_type = st.selectbox("Iris Type", iris_types)
    
    translate = st.checkbox("Translate to Indonesia")

    display_iris_info(iris_type, translate)

if (selected == 'Data Distribution'):
    st.header("Data Distribution")
    scatter_plot(df)
    
if (selected == 'Relation'):
    st.title('Relations')
    heatmap(df)

if (selected == 'Composition & Comparison'):
    st.title('Composition')
    compositionAndComparison(df)

if (selected == 'Predict'):
    st.title('Let\'s Predict!')
    predict()

if (selected == 'Clustering'):
    st.title('Clustering!')
    clustering(df)