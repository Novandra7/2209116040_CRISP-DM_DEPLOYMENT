import streamlit as st
import seaborn as sns
from googletrans import Translator
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def translate_text(text, target_language='id'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text

def translate_list_of_texts(text_list, target_language='id'):
    translator = Translator()
    translated_texts = [translator.translate(text, dest=target_language).text for text in text_list]
    return translated_texts

def display_iris_info(iris_type, translate):
    if iris_type == "Setosa":
        st.subheader("Setosa Iris")
        setosa_info = [
            "- Setosa irises are known for their distinctive appearance, with short, sturdy stems and showy flowers.",
            "- They typically have narrow leaves and bloom in various shades of white, pink, and lavender.",
            "- Setosa irises are native to North America and are well-suited to cooler climates.",
            ]
        if translate:
            setosa_info = translate_list_of_texts(setosa_info)
        st.markdown("\n".join(setosa_info))
    elif iris_type == "Versicolor":
        st.subheader("Versicolor Iris")
        versicolor_info = [
            "- Versicolor irises are characterized by their medium-sized flowers and broader range of colors compared to other iris species.",
            "- They often feature intricate patterns and combinations of blue, purple, violet, and white.",
            "- Versicolor irises are commonly found in wetlands and along the edges of ponds and streams in North America.",
            ]
        if translate:
            versicolor_info = translate_list_of_texts(versicolor_info)
        st.markdown("\n".join(versicolor_info))
    elif iris_type == "Virginica":
        st.subheader("Virginica Iris")
        virginica_info = [
            "- Virginica irises are tall and elegant, with long, slender leaves and striking flowers.",
            "- They typically bloom in shades of blue, purple, and white, with intricate veining and ruffling on the petals.",
            "- Virginica irises are native to wetland areas in eastern North America and are known for their resilience and adaptability.",
            ]
        if translate:
            virginica_info = translate_list_of_texts(virginica_info)
        st.markdown("\n".join(virginica_info))

def scatter_plot(df):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    class_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }
    # Membuat scatter plot
    for class_label in df['class'].unique():
        class_df = df[df['class'] == class_label]
        ax.scatter(class_df['petal_area'], class_df['sepal_area'], cmap='viridis', label=class_names[class_label])

    ax.set_xlabel('Petal Area')
    ax.set_ylabel('Sepal Area')
    ax.set_title("distribution of data from the iris data set")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    translate = st.checkbox("Translate to Indonesia")
    text = ["The scatter plot above shows the distribution or spread of data in this dataset with petal width as the X axis and sepal width as the Y axis.\n",
            "This shows that if the petal width is small it can be concluded that it is iris setosa, and if the petal width is large it is iris virginica, and among them is iris versicolor"]
    if translate:
        translated_text = translate_list_of_texts(text)
        if translated_text:
            text = translated_text
    st.markdown("\n".join(text))


def heatmap(df):
    df2 = df.drop(['class'],axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation between Numerical Features')
    st.pyplot(fig)
    translate = st.checkbox("Translate to Indonesia")
    text = 'As you can see, the heat map diagram above shows the correlation between all the columns in this dataset, meaning that the higher the value, the closer the relationship between the columns.'
    if translate:
        translated_text = translate_text(text)
        if translated_text:
            text = translated_text
    st.markdown(text)


def compositionAndComparison (df):
# Hitung rata-rata fitur untuk setiap kelas
    df['class'].replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)
    class_composition = df.groupby('class').mean()
    # Plot komposisi kelas
    plt.figure(figsize=(10, 6))
    sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu')
    plt.title('Composition for each class')
    plt.xlabel('Class')
    plt.ylabel('Feature')
    st.pyplot(plt)
    translate = st.checkbox("Translate to Indonesia")
    text = 'As you can see the bar plot above shows the composition of a class which is taken from the average of each existing feature (column) and there is also a comparison of each feature used.'
    if translate:
        translated_text = translate_text(text)
        if translated_text:
            text = translated_text
    st.markdown(text)

def predict ():
    petalArea = st.number_input('Petal Area')
    petalWidth = st.number_input('Petal Width')
    petalLength = st.number_input('Petal Length')
    sepalArea = st.number_input('Sepal Area')
    sepalWidth = st.number_input('Sepal Width')
    sepalLength = st.number_input('Sepal Length')
    button = st.button('Predict')
    data = pd.DataFrame({
        'sepallength' : [sepalLength],
        'sepalwidth' : [sepalWidth],
        'petallength' : [petalLength],
        'petalwidth' : [petalWidth],
        'petal_area' : [petalArea],
        'sepal_area' : [sepalArea]
    })
    st.write(data)
    if (button):
        with open('gnb.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        predicted = loaded_model.predict(data)
        if (predicted[0] == 0):
            st.image('img/setosa.png', caption='Iris Setosa', width=300)
        elif (predicted[0] == 1):
            st.image('img/versicolor.png', caption='Iris Versicolor', width=300)
        elif (predicted[0] == 2):
            st.image('img/virginica.png', caption='Iris Virginica', width=300)
        else :
            st.error('Not Defined')

def clustering (df):
    klasifikasi(df)
    petalArea = st.number_input('Petal Area')
    petalWidth = st.number_input('Petal Width')
    petalLength = st.number_input('Petal Length')
    sepalArea = st.number_input('Sepal Area')
    sepalWidth = st.number_input('Sepal Width')
    sepalLength = st.number_input('Sepal Length')
    button = st.button('Clust!')
    if (button):
        data = pd.DataFrame({
            'sepallength' : [sepalLength],
            'sepalwidth' : [sepalWidth],
            'petallength' : [petalLength],
            'petalwidth' : [petalWidth],
            'petalarea' : [petalArea],
            'sepalarea' : [sepalArea]
        })
        with open('kmeans.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        predicted = loaded_model.predict(data)
        print(predicted)
        if (predicted[0] == 0):
            st.image('img/setosa.png', caption='Iris Setosa', width=300)
        elif (predicted[0] == 1):
            st.image('img/versicolor.png', caption='Iris Versicolor', width=300)
        elif (predicted[0] == 2):
            st.image('img/virginica.png', caption='Iris Virginica', width=300)
        else :
            st.error('Not Defined')

def klasifikasi(df):
    x_final = df.drop("class", axis=1)
    scaler = MinMaxScaler()
    x_final_norm = scaler.fit_transform(x_final)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_final_norm)
    kmeans_clust = kmeans.predict(x_final_norm)
    x_final = pd.DataFrame(x_final).reset_index(drop=True)
    kmeans_col = pd.DataFrame(kmeans_clust, columns=["kmeans_cluster"])
    combined_data_assoc = pd.concat([x_final, kmeans_col], axis=1)
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_data_assoc['petal_area'], combined_data_assoc['sepal_area'], c=combined_data_assoc["kmeans_cluster"], cmap='viridis')
    plt.xlabel('Petal Area')
    plt.ylabel('Sepal Area')
    plt.title('Scatter Plot of K-Means Clustering')
    plt.colorbar(label='K-Means Cluster')
    plt.grid(True)
    st.pyplot(plt)