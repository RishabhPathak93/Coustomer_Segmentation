# data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handling missing values
    df.dropna(inplace=True)

    # One-hot encoding categorical features
    df = pd.get_dummies(df, columns=['Gender', 'Marital Status', 'Occupation', 'Location', 'Favorite Category'], drop_first=True)

    # Scaling numeric features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

    return df, df_scaled

if __name__ == "__main__":
    df, df_scaled = load_and_preprocess_data(r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\data\synthetic_data.csv')
    print("Data preprocessing complete!")
