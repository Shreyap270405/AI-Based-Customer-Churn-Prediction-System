# AI-Based Customer Churn Prediction for OTT Platforms (Netflix) using Big Data Technologies

## Overview
This project predicts whether a customer is likely to cancel an OTT subscription (churn) using Machine Learning and Big Data Technologies.

The system analyzes user behavior such as subscription duration, monthly fee, watch time, device type, payment method, and customer support interactions to predict churn risk.

## Objectives
- Predict customer churn using AI/ML
- Use Apache Spark for scalable big data preprocessing
- Build a churn prediction model using Random Forest
- Develop an interactive Streamlit application for real-time predictions

## Technologies Used
- Python
- Apache Spark (PySpark)
- Pandas
- Scikit-learn
- Streamlit

## Features Used
- Subscription Duration
- Monthly Fee
- Watch Time
- Device Type
- Payment Method
- Customer Support Usage

## Workflow
1. Load subscription churn dataset
2. Preprocess data using Apache Spark
3. Perform feature engineering
4. Convert Spark DataFrame to Pandas
5. Train machine learning model
6. Evaluate model accuracy
7. Deploy prediction interface using Streamlit

## Machine Learning Model
Algorithm used:
- Random Forest Classifier

Evaluation:
- Accuracy Score
- Confusion Matrix
- Prediction Probability

## Project Structure
```text
project/
│
├── data/
│   └── subscription_churn.csv
│
├── app.py
├── model_training.py
├── spark_processing.py
├── requirements.txt
└── README.md
```

## Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Streamlit App

```bash
python -m streamlit run app.py
```

## Example Prediction

High churn-risk customer:
- Low subscription duration
- High monthly fee
- Low watch time
- No customer support satisfaction

Prediction:
- Likely to Churn

## Big Data Component
Apache Spark is used for:
- Large-scale data preprocessing
- Distributed processing
- Scalable feature transformation

This satisfies the Big Data Technologies component of the project.

## Real-World Impact
This system can help OTT platforms:
- Identify at-risk subscribers
- Improve retention strategies
- Reduce revenue loss

## Future Scope
- Use deep learning models
- Real-time streaming data prediction
- Personalized retention recommendations
- Integration with dashboards

## Author
Big Data Technologies Mini Project
AI-Based Customer Churn Prediction for OTT Platforms
