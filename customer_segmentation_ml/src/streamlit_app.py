# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data and models
df = pd.read_csv(r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\data\synthetic_data.csv')
kmeans = joblib.load(r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\models\kmeans_model.pkl')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
age = st.sidebar.slider('Age', 18, 70)
income = st.sidebar.slider('Annual Income (k$)', 10, 150)
spending_score = st.sidebar.slider('Spending Score (1-100)', 1, 100)

# Prepare input for model
input_data = pd.DataFrame({'Age': [age], 'Annual Income (k$)': [income], 'Spending Score (1-100)': [spending_score]})
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Predict cluster
cluster = kmeans.predict(input_data_scaled)

# Display results
st.write(f'The customer belongs to cluster {cluster[0]}')

# Plot data and clusters
st.write('Customer Segmentation Data Overview')
st.dataframe(df.head())

plt.scatter(df['Age'], df['Annual Income (k$)'], c=kmeans.labels_)
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
st.pyplot(plt)
