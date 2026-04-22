import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    print("Reading data from", input_path)
    df = pd.read_csv(input_path)
    
    # Define columns
    categorical_cols = ["Device_Type", "Payment_Method"]
    
    # Initialize LabelEncoders
    le_device = LabelEncoder()
    le_payment = LabelEncoder()
    
    print("Applying Label Encoding...")
    df['Device_Type_numeric'] = le_device.fit_transform(df['Device_Type'])
    df['Payment_Method_numeric'] = le_payment.fit_transform(df['Payment_Method'])
    
    # Drop original categorical columns
    final_df = df.drop(columns=categorical_cols)
    
    print(f"Writing processed data to {output_path}...")
    final_df.to_csv(output_path, index=False)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data("ott_data.csv", "processed_data.csv")
