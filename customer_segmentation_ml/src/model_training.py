# model_training.py
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from tqdm import tqdm

def train_kmeans(df_scaled, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_scaled)
    joblib.dump(kmeans, r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\models\kmeans_model.pkl')
    print(f"K-Means model with {n_clusters} clusters saved.")

def train_dbscan(df_scaled):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(df_scaled)
    joblib.dump(dbscan, r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\models\dbscan_model.pkl')
    print("DBSCAN model saved.")

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\data\synthetic_data.csv')
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

    # Train both models
    train_kmeans(df_scaled)
    train_dbscan(df_scaled)
