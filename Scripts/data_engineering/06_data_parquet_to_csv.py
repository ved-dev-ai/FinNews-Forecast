"""
Convert categorical stocks data from parquet to individual CSV files by category.
"""

import pandas as pd
import os

# Load the parquet dataset
df = pd.read_parquet('../../Datasets/categorical_stocks_data')

# Create output directory if it doesn't exist
output_dir = '../../Datasets/category_csvs'
os.makedirs(output_dir, exist_ok=True)

# Split by category and save each as CSV
for category in df['category'].unique():
    category_df = df[df['category'] == category]
    filename = f"{output_dir}/{category}_stocks_data.csv"
    category_df.to_csv(filename, index=False)
    print(f"Saved {category}: {len(category_df)} rows to {filename}")

print(f"\nTotal categories saved: {df['category'].nunique()}")
