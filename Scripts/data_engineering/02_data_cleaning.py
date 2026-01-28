"""
Data Cleaning Script for FinNews-Forecast Project

This script performs data cleaning operations on both news data and stock price data:
1. Cleans and filters news data for S&P 500 stocks
2. Processes stock price historical data
3. Saves cleaned data to S3/MinIO in parquet format

Author: FinNews-Forecast Team
Date: January 2026
"""

import pandas as pd
import requests
from io import StringIO
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType, 
    DoubleType, LongType
)


def create_spark_session():
    """
    Create and configure Spark session with S3/MinIO connectivity.
    
    Returns:
        SparkSession: Configured Spark session
    """
    spark = SparkSession.builder \
        .appName("Data Cleaning") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .getOrCreate()
    
    print(f"Spark Application Name: {spark.sparkContext.appName}")
    print(f"Spark Version: {spark.sparkContext.version}")
    print(f"Python Version: {spark.sparkContext.pythonVer}")
    print(f"Master: {spark.sparkContext.master}")
    
    return spark


def get_sp500_symbols():
    """
    Fetch S&P 500 stock symbols from Wikipedia.
    
    Returns:
        list: List of S&P 500 stock symbols
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # Fetch and parse the Wikipedia page
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    
    # Extract stock symbols from the first table
    sp500_table = tables[0]
    stock_list = sp500_table['Symbol'].to_list()
    
    print(f"Retrieved {len(stock_list)} S&P 500 stock symbols")
    return stock_list


def clean_news_data(spark, bucket='fnf-bucket'):
    """
    Clean and process news data.
    
    This function:
    - Loads raw news data from S3
    - Filters for S&P 500 stocks only
    - Drops unnecessary columns
    - Handles missing values
    - Saves cleaned data back to S3 in parquet format
    
    Args:
        spark (SparkSession): Active Spark session
        bucket (str): S3/MinIO bucket name
    """
    print("\n" + "="*50)
    print("Starting News Data Cleaning")
    print("="*50)
    
    # Define schema for news data
    data_schema = StructType([
        StructField("No", StringType(), True),
        StructField("Date", StringType(), True),
        StructField("Article_title", StringType(), True),
        StructField("Stock_symbol", StringType(), True),
        StructField("Url", StringType(), True),
        StructField("Publisher", StringType(), True),
        StructField("Author", StringType(), True),
        StructField("Article", StringType(), True),
        StructField("Lsa_summary", StringType(), True),
        StructField("Luhn_summary", StringType(), True),
        StructField("Textrank_summary", StringType(), True),
        StructField("Lexrank_summary", StringType(), True)
    ])
    
    # Load news data from S3
    object_path = 'bronze/stock_news/nasdaq_exteral_data.csv'
    s3a_path = f's3a://{bucket}/{object_path}'
    
    print(f"Loading news data from: {s3a_path}")
    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("escape", "\"") \
        .option("multiline", "true") \
        .schema(data_schema) \
        .load(s3a_path)
    
    print(f"Initial record count: {df.count()}")
    print("\nInitial Schema:")
    df.printSchema()
    
    # Drop unnecessary columns
    df = df.drop(
        F.col('Publisher'), 
        F.col('Author'), 
        F.col('Luhn_summary'), 
        F.col('Textrank_summary'), 
        F.col('Lexrank_summary')
    )
    
    print("\nSchema after dropping columns:")
    df.printSchema()
    
    # Get S&P 500 stock symbols
    stock_list = get_sp500_symbols()
    
    # Filter for S&P 500 stocks only
    sp500_df = df.where(F.col('Stock_symbol').isin(stock_list))
    print(f"Records after S&P 500 filter: {sp500_df.count()}")
    
    # Cast data types appropriately
    sp500_df = sp500_df.withColumn(
        "No", F.col("No").cast("integer")
    ).withColumn(
        "Date", F.to_date(F.col("Date"), "yyyy-MM-dd HH:mm:ss z")
    )
    
    # Remove records with null values
    df_clean = sp500_df.dropna()
    print(f"Records after dropping nulls: {df_clean.count()}")
    
    print("\nFinal Schema:")
    df_clean.printSchema()
    
    # Save cleaned data to S3 in parquet format, partitioned by stock symbol
    output_path = f's3a://{bucket}/silver/news_data_clean'
    print(f"\nSaving cleaned news data to: {output_path}")
    
    df_clean.write \
        .mode('overwrite') \
        .option('compression', 'snappy') \
        .partitionBy('Stock_symbol') \
        .parquet(output_path)
    
    print("News data cleaning completed successfully!")


def clean_stock_price_data(spark, bucket='fnf-bucket'):
    """
    Clean and process stock price data.
    
    This function:
    - Loads raw stock price data from S3
    - Normalizes column names
    - Casts data types appropriately
    - Extracts stock symbol from filename
    - Saves cleaned data back to S3 in parquet format
    
    Args:
        spark (SparkSession): Active Spark session
        bucket (str): S3/MinIO bucket name
    """
    print("\n" + "="*50)
    print("Starting Stock Price Data Cleaning")
    print("="*50)
    
    # Define input path (all CSV files in the directory)
    input_path = f"s3a://{bucket}/bronze/stock_price/full_history/*.csv"
    
    print(f"Loading stock price data from: {input_path}")
    
    # Read CSV files with header and infer schema
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(input_path)
    
    # Normalize column names (handle case sensitivity and spaces)
    for col_name in df.columns:
        df = df.withColumnRenamed(col_name, col_name.strip())
    
    # Select and cast columns to appropriate types
    df = df.select(
        F.to_date(F.col("Date"), "yyyy-MM-dd").alias("Date"),
        F.col("Open").cast(DoubleType()).alias("Open"),
        F.col("High").cast(DoubleType()).alias("High"),
        F.col("Low").cast(DoubleType()).alias("Low"),
        F.col("Close").cast(DoubleType()).alias("Close"),
        F.col("Adj Close").cast(DoubleType()).alias("Adj Close"),
        F.col("Volume").cast(LongType()).alias("Volume")
    )
    
    # Extract stock symbol from filename and add as column
    # Pattern matches filename without .csv extension
    df = df.withColumn(
        "Stock_symbol",
        F.upper(
            F.regexp_extract(
                F.input_file_name(),
                r"([^/]+)\.csv$",  # Extract filename without extension
                1
            )
        )
    )
    
    print(f"Total records loaded: {df.count()}")
    print("\nFinal Schema:")
    df.printSchema()
    
    # Save to parquet format, partitioned by stock symbol
    output_path = f"s3a://{bucket}/silver/stock_price_data"
    print(f"\nSaving cleaned stock price data to: {output_path}")
    
    df.write \
        .mode("overwrite") \
        .partitionBy("Stock_symbol") \
        .parquet(output_path)
    
    print("Stock price data cleaning completed successfully!")


def main():
    """
    Main execution function.
    
    Orchestrates the entire data cleaning pipeline:
    1. Creates Spark session
    2. Cleans news data
    3. Cleans stock price data
    4. Closes Spark session
    """
    print("\n" + "="*50)
    print("FinNews-Forecast Data Cleaning Pipeline")
    print("="*50 + "\n")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Clean news data
        clean_news_data(spark)
        
        # Clean stock price data
        clean_stock_price_data(spark)
        
        print("\n" + "="*50)
        print("All data cleaning operations completed successfully!")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\nError occurred during data cleaning: {str(e)}")
        raise
    
    finally:
        # Stop Spark session
        print("Stopping Spark session...")
        spark.stop()
        print("Spark session stopped.")


if __name__ == "__main__":
    main()
