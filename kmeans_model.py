
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

def find_optimal_k(data, max_k=10):
    sse = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        model.fit(data)
        sse.append(model.inertia_)
    logging.info("Elbow Method: SSE values computed.")
    return sse

def silhouette_scores(data, max_k=10):
    scores = []
    for k in range(2, max_k + 1):  # silhouette score not defined for k=1
        model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
    logging.info("Silhouette scores computed.")
    return scores

def train_kmeans(data, k=6):
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    model.fit(data)
    labels = model.predict(data)
    logging.info(f"KMeans model trained with k={k}.")
    return model, labels
