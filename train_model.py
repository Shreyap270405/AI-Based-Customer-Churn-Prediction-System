import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_churn_model(data_path, model_output_path):
    """
    Trains a Machine Learning model to predict OTT customer churn.
    Uses the cleaned Pandas DataFrame exported from the PySpark pipeline.
    """
    print(f"📂 Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Separate Features (X) and Target Variable (y)
    # User_ID is dropped because it's an identifier, not a predictive feature
    X = df.drop(columns=["User_ID", "Churn"])
    y = df["Churn"]
    
    # 2. Train-Test Split
    # We use 80% of the data for training and 20% for testing.
    # stratify=y ensures the train and test sets have the same proportion of churned vs non-churned users.
    print("✂️ Splitting data into Training and Testing sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Handle Class Imbalance & Initialize Random Forest
    # In churn prediction, non-churners often outnumber churners (class imbalance).
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies.
    print("🌲 Initializing Random Forest Classifier (Handling Class Imbalance)...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    )
    
    # Train the model
    print("⚙️ Training the model...")
    rf_model.fit(X_train, y_train)
    
    # 4. Make Predictions
    print("🔮 Generating predictions on the test set...")
    y_pred = rf_model.predict(X_test)
    
    # 5. Probability Prediction
    # Instead of just 1 (Churn) or 0 (Stay), get the exact probability % of churning
    y_prob = rf_model.predict_proba(X_test)[:, 1] # Probability of class 1 (Churn)
    
    # Display the first 5 probability predictions
    print("\nSample Probability Predictions (Top 5 users in test set):")
    for i in range(5):
        print(f"User {i+1} -> Churn Probability: {y_prob[i]*100:.2f}% | Final Prediction: {'Churn' if y_pred[i] == 1 else 'Stay'}")
    
    # 6. Evaluate the Model
    print("\n📊 Model Evaluation Metrics:")
    
    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy * 100:.2f}%\n")
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(
        cm, 
        columns=['Predicted: Stay (0)', 'Predicted: Churn (1)'], 
        index=['Actual: Stay (0)', 'Actual: Churn (1)']
    ))
    
    # Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the trained model for deployment (e.g., in a Streamlit app)
    print(f"💾 Saving trained model to {model_output_path}...")
    joblib.dump(rf_model, model_output_path)
    print("✅ Machine Learning Pipeline Complete!")

if __name__ == "__main__":
    # Ensure this matches the output filename from the PySpark step
    train_churn_model("processed_data_spark.csv", "churn_model.pkl")
