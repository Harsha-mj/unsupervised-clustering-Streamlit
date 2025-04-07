
import streamlit as st
import pandas as pd
from data_preprocessing import load_data, preprocess_data
from kmeans_model import find_optimal_k, train_kmeans, silhouette_scores
from utils import plot_elbow, plot_silhouette_scores, show_correlation_heatmap

st.set_page_config(page_title="KMeans Clustering App", layout="centered")
st.title("📊 Unsupervised Clustering with KMeans")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load and show raw data
        df = load_data(uploaded_file)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Show correlation heatmap
        st.subheader("Correlation Heatmap")
        feature = df.select_dtypes(include=['int64', 'float64']).copy()
        show_correlation_heatmap(feature, figsize=(8, 6))

        # Preprocess and apply PCA
        reduced_features, numerical_df = preprocess_data(df)

        # Elbow method
        st.subheader("🔍 Elbow Method for Optimal k")
        max_k = st.slider("Select max value for k", min_value=5, max_value=15, value=10)
        sse = find_optimal_k(reduced_features, max_k)
        plot_elbow(sse, max_k,, figsize=(8, 6))

        # Silhouette method
        st.subheader("📏 Silhouette Score Method for Optimal k")
        silhouette_scores_list = silhouette_scores(reduced_features, max_k)
        plot_silhouette_scores(silhouette_scores_list, max_k,, figsize=(8, 6))

        # User-selected k
        st.subheader("🎯 Select Number of Clusters for Final Model")
        selected_k = st.slider("Choose final value of k", min_value=2, max_value=max_k, value=3)

        # Train final model
        model, labels = train_kmeans(reduced_features, k=selected_k)
        df['Cluster'] = labels
        st.write(df[['Cluster']].value_counts())

        st.subheader("Clustered Data Preview")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")
