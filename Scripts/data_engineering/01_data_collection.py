"""
Data Collection Script for FinNews-Forecast

This script collects financial data from Hugging Face datasets and stores it in MinIO S3 storage.
It downloads stock news data, stock price data, and S&P 500 company information, then processes
and uploads them to designated S3 buckets for further analysis.

Author: FinNews-Forecast Team
Date: January 2026
"""

import os
from io import BytesIO, StringIO
import zipfile
import boto3
import pandas as pd
import requests
from minio import Minio
from huggingface_hub import list_repo_files
from tqdm import tqdm


# ================================
# Configuration
# ================================

# Hugging Face dataset configuration
DATASET_ID = "Zihan1004/FNSPID"

# S3/MinIO bucket configuration
S3_BUCKET = "fnf-bucket"

# MinIO connection settings
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"


# ================================
# S3 Client Initialization
# ================================

def initialize_s3_client():
    """
    Initialize and return a boto3 S3 client configured for MinIO.
    
    Returns:
        boto3.client: Configured S3 client
    """
    s3_client = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        use_ssl=False
    )
    return s3_client


# ================================
# Bucket Management
# ================================

def create_bucket_if_not_exists(s3_client, bucket_name):
    """
    Check if S3 bucket exists, create it if it doesn't.
    
    Args:
        s3_client: Boto3 S3 client instance
        bucket_name (str): Name of the bucket to create
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists")
    except Exception:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"Created bucket '{bucket_name}'")


# ================================
# Data Download Functions
# ================================

def get_news_data(s3_client):
    """
    Download stock news data from Hugging Face and upload to S3.
    
    Fetches the NASDAQ external news data CSV file and stores it in the bronze layer
    of the data lake for raw data storage.
    
    Args:
        s3_client: Boto3 S3 client instance
    """
    print("\nDownloading stock news data...")
    hf_url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/Stock_news/nasdaq_exteral_data.csv"
    
    try:
        response = requests.get(hf_url, stream=True)
        response.raise_for_status()
        
        s3_key = "bronze/stock_news/nasdaq_exteral_data.csv"
        
        s3_client.upload_fileobj(
            response.raw,
            S3_BUCKET,
            s3_key
        )
        print("News data upload successful!")
        
    except Exception as e:
        print(f"Error uploading news data to S3: {e}")
        raise


def get_stocks_data(s3_client):
    """
    Download stock price data from Hugging Face and upload to S3.
    
    Fetches the complete historical stock price data as a ZIP file and stores it
    in the bronze layer for later extraction and processing.
    
    Args:
        s3_client: Boto3 S3 client instance
    """
    print("\nDownloading stock price data...")
    hf_url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main/Stock_price/full_history.zip"
    
    try:
        response = requests.get(hf_url, stream=True)
        response.raise_for_status()
        
        s3_key = "bronze/stock_price/full_history.zip"
        
        s3_client.upload_fileobj(
            response.raw,
            S3_BUCKET,
            s3_key
        )
        print("Stock price data upload successful!")
        
    except Exception as e:
        print(f"Error uploading stock price data to S3: {e}")
        raise


# ================================
# S&P 500 Stock List Retrieval
# ================================

def get_sp500_stock_list():
    """
    Scrape the current list of S&P 500 stock symbols from Wikipedia.
    
    Returns:
        list: List of stock ticker symbols in the S&P 500 index
    """
    print("\nFetching S&P 500 stock list from Wikipedia...")
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML tables
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]
        
        # Extract stock symbols
        stock_list = sp500_table['Symbol'].to_list()
        
        print(f"Retrieved {len(stock_list)} S&P 500 stock symbols")
        return stock_list
        
    except Exception as e:
        print(f"Error fetching S&P 500 stock list: {e}")
        raise


# ================================
# Data Extraction Functions
# ================================

def extract_zip(s3_client, stock_list):
    """
    Extract stock price CSV files from ZIP archive and upload to S3.
    
    This function reads the ZIP file from S3, extracts only the CSV files for stocks
    in the S&P 500 list, and uploads them individually to the bronze layer.
    
    Args:
        s3_client: Boto3 S3 client instance
        stock_list (list): List of stock symbols to filter and extract
    """
    print("\nExtracting stock price data from ZIP archive...")
    
    # Convert to set for O(1) lookup performance
    stock_set = set(stock_list)
    
    try:
        # Download ZIP file from S3
        response = s3_client.get_object(
            Bucket=S3_BUCKET,
            Key="bronze/stock_price/full_history.zip"
        )
        zip_bytes = BytesIO(response['Body'].read())
        
        # Process ZIP file
        with zipfile.ZipFile(zip_bytes, "r") as z:
            # Get list of CSV files, excluding MacOS metadata
            file_list = [
                name for name in z.namelist()
                if "__MACOSX" not in name and name.endswith(".csv")
            ]
            
            extracted_count = 0
            
            # Extract and upload relevant files
            for name in tqdm(file_list, desc="Extracting ZIP", unit="file"):
                # Extract stock symbol from filename
                stock_symbol = os.path.splitext(os.path.basename(name))[0].upper()
                
                # Skip files not in S&P 500 list
                if stock_symbol not in stock_set:
                    continue
                
                # Read file data from ZIP
                file_data = z.read(name)
                
                # Upload to S3
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=f"bronze/stock_price/{name}",
                    Body=BytesIO(file_data),
                    ContentLength=len(file_data)
                )
                
                extracted_count += 1
            
            print(f"Extracted {extracted_count} stock price files to S3")
            
    except Exception as e:
        print(f"Error extracting ZIP file: {e}")
        raise


# ================================
# Main Execution
# ================================

def main():
    """
    Main function to orchestrate the data collection pipeline.
    
    This function:
    1. Initializes S3 client and creates necessary buckets
    2. Downloads news and stock price data from Hugging Face
    3. Retrieves S&P 500 stock list
    4. Extracts and uploads individual stock price files
    """
    print("=" * 60)
    print("Starting Data Collection Pipeline")
    print("=" * 60)
    
    try:
        # Initialize S3 client
        s3_client = initialize_s3_client()
        print(" S3 client initialized")
        
        # Create bucket if needed
        create_bucket_if_not_exists(s3_client, S3_BUCKET)
        
        # Download news data
        get_news_data(s3_client)
        
        # Download stock price data
        get_stocks_data(s3_client)
        
        # Get S&P 500 stock list
        stock_list = get_sp500_stock_list()
        
        # Extract and upload stock price files
        extract_zip(s3_client, stock_list)
        
        print("\n" + "=" * 60)
        print("Data Collection Pipeline Completed Successfully!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"Pipeline failed with error: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
