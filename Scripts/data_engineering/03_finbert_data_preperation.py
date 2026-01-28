"""
FinBERT Data Preparation Script

This script prepares news data for FinBERT sentiment analysis by:
1. Reading cleaned news data from MinIO S3 storage
2. Selecting relevant columns (article number and LSA summary)
3. Writing the prepared data back to S3 for FinBERT processing

Author: Data Engineering Team
Date: 2026-01-28
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def create_spark_session():
    """
    Create and configure a Spark session with MinIO S3 connectivity.
    
    Returns:
        SparkSession: Configured Spark session object
    """
    spark = SparkSession.builder \
        .appName("FinBERT Data Preparation") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .getOrCreate()
    
    return spark


def prepare_finbert_data(spark, input_path, output_path):
    """
    Prepare data for FinBERT sentiment analysis.
    
    Args:
        spark (SparkSession): Active Spark session
        input_path (str): S3 path to cleaned news data
        output_path (str): S3 path to write prepared data
        
    Returns:
        int: Number of records processed
    """
    # Read cleaned news data from S3
    print(f"Reading data from {input_path}...")
    df = spark.read.parquet(input_path)
    
    # Display schema for verification
    print("\nData Schema:")
    df.printSchema()
    
    # Select only the columns needed for FinBERT processing
    # No: Article identifier
    # Lsa_summary: LSA-generated summary of the news article
    df_finbert = df.select(
        F.col("No"),
        F.col("Lsa_summary")
    )
    
    # Write prepared data to S3 in Parquet format with Snappy compression
    print(f"\nWriting prepared data to {output_path}...")
    df_finbert.write \
        .mode('overwrite') \
        .option('compression', 'snappy') \
        .parquet(output_path)
    
    # Get and return the count of processed records
    record_count = df_finbert.count()
    print(f"\nTotal records processed: {record_count}")
    
    return record_count


def main():
    """
    Main execution function.
    """
    # S3 paths
    INPUT_PATH = 's3a://fnf-bucket/silver/news_data_clean'
    OUTPUT_PATH = 's3a://fnf-bucket/silver/data_for_finbert'
    
    # Initialize Spark session
    print("Initializing Spark session...")
    spark = create_spark_session()
    
    try:
        # Prepare data for FinBERT
        record_count = prepare_finbert_data(spark, INPUT_PATH, OUTPUT_PATH)
        print(f"\nData preparation completed successfully!")
        print(f"Prepared {record_count} records for FinBERT sentiment analysis.")
        
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        raise
        
    finally:
        # Stop Spark session
        print("\nStopping Spark session...")
        spark.stop()
        print("Done!")


if __name__ == "__main__":
    main()
