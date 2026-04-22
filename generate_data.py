import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of users
num_users = 10000

# Generate features
user_ids = np.arange(1, num_users + 1)
subscription_duration_months = np.random.randint(1, 60, num_users)
monthly_fee = np.random.choice([9.99, 14.99, 19.99], num_users)
watch_time_hours = np.random.normal(loc=20, scale=10, size=num_users)
watch_time_hours = np.clip(watch_time_hours, 0, None)  # No negative watch time
device_type = np.random.choice(['Smart TV', 'Mobile', 'Tablet', 'PC'], num_users)
payment_method = np.random.choice(['Credit Card', 'PayPal', 'Carrier Billing'], num_users)
customer_support_interactions = np.random.poisson(lam=1.5, size=num_users)

# Generate Churn labels based on some rules with noise
# Higher churn if: low watch time, high fee, high support interactions, low duration
churn_prob = (
    (watch_time_hours < 5).astype(int) * 0.3 + 
    (monthly_fee == 19.99).astype(int) * 0.1 + 
    (customer_support_interactions > 3).astype(int) * 0.2 + 
    (subscription_duration_months < 6).astype(int) * 0.2
)

# Add some randomness
churn_prob += np.random.uniform(0, 0.2, num_users)

# Threshold to get 1 or 0
churn = (churn_prob > 0.45).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'User_ID': user_ids,
    'Subscription_Duration_Months': subscription_duration_months,
    'Monthly_Fee': monthly_fee,
    'Watch_Time_Hours': watch_time_hours,
    'Device_Type': device_type,
    'Payment_Method': payment_method,
    'Customer_Support_Interactions': customer_support_interactions,
    'Churn': churn
})

# Save to CSV
df.to_csv('ott_data.csv', index=False)
print("Generated ott_data.csv successfully with", num_users, "records.")
