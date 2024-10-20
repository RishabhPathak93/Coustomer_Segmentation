import os

# Define directory structure
directories = [
    "customer_segmentation_ml/data",
    "customer_segmentation_ml/models",
    "customer_segmentation_ml/src"
]

# Define empty files to be created
files = {
    "customer_segmentation_ml/data/synthetic_data.csv": "",
    "customer_segmentation_ml/models/kmeans_model.pkl": "",
    "customer_segmentation_ml/models/dbscan_model.pkl": "",
    "customer_segmentation_ml/src/data_preprocessing.py": "# Data Preprocessing Script\n\nif __name__ == '__main__':\n    pass",
    "customer_segmentation_ml/src/data_visualization.py": "# Data Visualization Script\n\nif __name__ == '__main__':\n    pass",
    "customer_segmentation_ml/src/model_training.py": "# Model Training Script\n\nif __name__ == '__main__':\n    pass",
    "customer_segmentation_ml/src/deep_learning_model.py": "# Deep Learning Model Script\n\nif __name__ == '__main__':\n    pass",
    "customer_segmentation_ml/src/streamlit_app.py": "# Streamlit App Script\n\nif __name__ == '__main__':\n    pass",
    "customer_segmentation_ml/requirements.txt": "streamlit\nscikit-learn\nmatplotlib\nseaborn\npandas\ntqdm\njoblib\ntensorflow",
    "customer_segmentation_ml/README.md": "# Customer Segmentation Using Machine Learning\n\nThis project aims to segment customers based on their banking behavior and preferences using clustering algorithms.\n\n## Directory Structure\n\n- `data/`: Contains the dataset\n- `models/`: Contains the saved models\n- `src/`: Contains source code for preprocessing, model training, and visualization\n\n## Requirements\n\nPlease install the required packages using:\n```\npip install -r requirements.txt\n```"
}

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Directory created: {directory}")

# Create files with the initial content
for file_path, content in files.items():
    with open(file_path, "w") as file:
        file.write(content)
    print(f"File created: {file_path}")
