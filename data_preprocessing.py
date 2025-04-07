
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

def load_data(filepath):
    try:
        if filepath.name.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.name.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    # Selecting numerical features only for clustering
    features = df.select_dtypes(include=['int64', 'float64']).copy()

    # Standardizing features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # PCA to reduce dimensionality to 3 components
    pca = PCA(n_components=3, random_state=42)
    reduced_features = pca.fit_transform(scaled_features)

    return reduced_features, features
