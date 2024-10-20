import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_data(df):
    # 1. Male vs Female Ratio (Gender Distribution)
    gender_counts = df['Gender'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightpink'])
    plt.title('Male vs Female Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
    plt.show()

    # 2. Correlation Heatmap for numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\RISHABH\OneDrive\Desktop\SIH1693\customer_segmentation_ml\data\synthetic_data.csv')
    visualize_data(df)
