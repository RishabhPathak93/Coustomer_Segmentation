# deep_learning_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import pandas as pd
import joblib

def build_and_train_dl_model(df_scaled, labels, input_dim):
    # Convert labels to one-hot encoding
    labels_one_hot = to_categorical(labels, num_classes=5)

    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # Assuming 5 clusters from K-Means

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(df_scaled, labels_one_hot, epochs=200, batch_size=150)


if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\data\synthetic_data.csv'
                     )
    kmeans = joblib.load( r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\models\kmeans_model.pkl')

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

    build_and_train_dl_model(df_scaled, kmeans.labels_, df_scaled.shape[1])
