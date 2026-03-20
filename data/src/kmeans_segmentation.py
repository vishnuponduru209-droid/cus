import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    X = df[['AnnualIncome', 'SpendingScore']].values
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def train_kmeans(X, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def evaluate(X, labels):
    return {
        "Silhouette": silhouette_score(X, labels),
        "DB Index": davies_bouldin_score(X, labels),
        "CH Index": calinski_harabasz_score(X, labels)
    }
