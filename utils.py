
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_elbow(sse, max_k=10):
    fig, ax = plt.subplots()
    ax.plot(range(1, max_k + 1), sse, marker='o')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('SSE')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)

def plot_silhouette_scores(scores, max_k=10):
    fig, ax = plt.subplots()
    ax.plot(range(2, max_k + 1), scores, marker='o', color='green')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for Various k')
    st.pyplot(fig)

def show_correlation_heatmap(df):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
