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
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
        logging.info("Data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    try:
        # Select only numeric columns
        features = df.select_dtypes(include=['int64', 'float64']).copy()

        # Drop rows with missing values
        features = features.dropna()

        if features.empty:
            raise ValueError("No numeric features found after preprocessing.")

        # Standardize the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # Apply PCA to reduce dimensions to 3 components
        pca = PCA(n_components=3, random_state=42)
        reduced_features = pca.fit_transform(scaled_features)

        logging.info("Preprocessing completed successfully.")
        return reduced_features, features

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

