import os
import sys

# Optional: Ensure PySpark uses the correct Python executable
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

def run_spark_pipeline(input_csv, output_csv):
    """
    A simple PySpark pipeline demonstrating scalable Big Data processing 
    for an OTT Churn Prediction project.
    """
    
    # 1. Initialize SparkSession (scalable processing entry point)
    print("🚀 Initializing SparkSession...")
    spark = SparkSession.builder \
        .appName("OTT_Churn_BigData_Processing") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
        
    # 2. Load the dataset (distributed Spark DataFrame)
    print(f"📂 Loading dataset: {input_csv}")
    # inferSchema=True automatically detects data types (Int, Double, String)
    df = spark.read.csv(input_csv, header=True, inferSchema=True)
    
    # Show initial data structure
    print("Initial Data Schema:")
    df.printSchema()
    
    # 3. Handle Missing Values
    # In a massive dataset, dropping or filling NA values across a cluster is done seamlessly by Spark
    print("🧹 Handling missing values...")
    # Fill missing numeric values with 0 (or you could use the mean)
    df = df.fillna({
        'Watch_Time_Hours': 0.0, 
        'Monthly_Fee': 0.0,
        'Customer_Support_Interactions': 0
    })
    # Drop rows where critical columns like User_ID or Churn are missing
    df = df.dropna(subset=['User_ID', 'Churn'])

    # 4. Feature Engineering
    # Creating a new feature: "Cost per Hour of Watch Time"
    print("⚙️ Performing Feature Engineering...")
    df = df.withColumn(
        "Cost_Per_Hour", 
        when(col("Watch_Time_Hours") > 0, col("Monthly_Fee") / col("Watch_Time_Hours")).otherwise(0.0)
    )
    
    # 5. Encode Categorical Columns
    print("🔠 Encoding Categorical Columns...")
    categorical_cols = ["Device_Type", "Payment_Method"]
    
    # StringIndexer converts string categories into numerical indices (e.g., 'Mobile' -> 0.0)
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
        for c in categorical_cols
    ]
    
    # OneHotEncoder transforms indices into binary vectors (ideal for ML models)
    encoders = [
        OneHotEncoder(inputCol=f"{c}_indexed", outputCol=f"{c}_vec")
        for c in categorical_cols
    ]
    
    # Build a Spark ML Pipeline to run transformations in sequence
    pipeline = Pipeline(stages=indexers + encoders)
    
    # Fit and transform the data across the cluster
    pipeline_model = pipeline.fit(df)
    processed_df = pipeline_model.transform(df)
    
    # Drop the intermediate string and indexed columns to keep it clean
    cols_to_drop = categorical_cols + [f"{c}_indexed" for c in categorical_cols]
    final_spark_df = processed_df.drop(*cols_to_drop)
    
    print("Preview of processed Spark DataFrame:")
    final_spark_df.show(5, truncate=False)
    
    # 6. Convert final Spark DataFrame to Pandas
    # This is done AFTER heavy distributed processing is complete, 
    # bringing the reduced/cleaned data back to the driver node for Scikit-Learn training.
    print("🔄 Converting Spark DataFrame to Pandas...")
    pandas_df = final_spark_df.toPandas()
    
    # Save the final Pandas DataFrame to a CSV
    print(f"💾 Saving processed data to {output_csv}")
    pandas_df.to_csv(output_csv, index=False)
    
    print("✅ PySpark Processing Complete!")
    spark.stop()

if __name__ == "__main__":
    # Ensure you have 'ott_data.csv' in the same directory before running
    run_spark_pipeline("ott_data.csv", "processed_data_spark.csv")
