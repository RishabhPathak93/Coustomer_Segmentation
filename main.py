import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Parameters
num_customers = 100000  # Number of customers

# Generate Customer IDs
customer_ids = np.arange(1, num_customers + 1)

# Generate Gender
genders = np.random.choice(['Male', 'Female'], size=num_customers)

# Generate Age
ages = np.random.randint(18, 71, size=num_customers)

# Generate Annual Income
annual_income = np.random.randint(15, 121, size=num_customers)

# Generate Spending Score
spending_score = np.random.randint(1, 101, size=num_customers)

# Generate additional features
marital_status = np.random.choice(['Single', 'Married', 'Divorced'], size=num_customers)
occupations = np.random.choice(['Student', 'Professional', 'Retired', 'Other'], size=num_customers)
locations = np.random.choice([
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata',
    'Pune', 'Jaipur', 'Surat', 'Kanpur', 'Nagpur', 'Lucknow', 'Visakhapatnam',
    'Bhopal', 'Patna', 'Vadodara', 'Indore', 'Coimbatore', 'Chandigarh', 'Agra',
    'Madurai', 'Nashik', 'Ranchi', 'Ghaziabad', 'Aurangabad', 'Allahabad', 'Varanasi',
    'Jodhpur', 'Amritsar', 'Faridabad', 'Meerut', 'Mysore', 'Raipur', 'Durgapur',
    'Kalyan', 'Thane', 'Srinagar', 'Dehradun', 'Patiala', 'Nagaland', 'Salem',
    'Udaipur', 'Kota', 'Warangal', 'Vellore', 'Tiruchirappalli', 'Siliguri',
    'Guntur', 'Nanded', 'Vijayawada', 'Bikaner', 'Ajmer', 'Jabalpur', 'Rourkela',
    'Bhubaneswar', 'Bhilai', 'Agartala', 'Imphal', 'Itanagar', 'Aizawl', 'Gangtok',
    'Shillong', 'Kohima', 'Dibrugarh', 'Guwahati', 'Tezpur', 'Jorhat'
], size=num_customers)
purchase_frequency = np.random.choice(['Daily', 'Weekly', 'Monthly'], size=num_customers)
loyalty_status = np.random.choice(['New', 'Returning', 'Loyal'], size=num_customers)
last_purchase_date = pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2024-01-01'), size=num_customers))
favorite_category = np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Furtinture', 'toys', 'eatables', 'others'], size=num_customers)

# Create a DataFrame
customer_data = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Annual Income (k$)': annual_income,
    'Spending Score (1-100)': spending_score,
    'Marital Status': marital_status,
    'Occupation': occupations,
    'Location': locations,
    'Purchase Frequency': purchase_frequency,
    'Loyalty Status': loyalty_status,
    'Last Purchase Date': last_purchase_date,
    'Favorite Category': favorite_category
})

# Save to CSV
customer_data.to_csv('synthetic_customer_data_with_indian_cities.csv', index=False)

print("Synthetic customer dataset with Indian cities created successfully!")
print(customer_data.head())  # Display the first few rows of the dataset
