"""
Merge Sentiment Data with Stock Price Data

This script performs the following operations:
1. Loads and aggregates sentiment data from FinBERT analysis
2. Merges sentiment data with cleaned news data
3. Joins sentiment data with stock price data
4. Categorizes stocks by industry sector
5. Outputs final categorized dataset for ML training

Author: FinNews-Forecast Team
Date: January 2026
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType


def create_spark_session():
    """
    Initialize and configure Spark session with MinIO S3 connectivity.
    
    Returns:
        SparkSession: Configured Spark session
    """
    spark = SparkSession.builder \
        .appName("Merge Sentiment Data") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "20") \
        .config("spark.sql.autoBroadcastJoinThreshold", -1) \
        .getOrCreate()
    
    # Set log level to reduce verbosity
    spark.sparkContext.setLogLevel("ERROR")
    
    return spark


def load_and_merge_news_sentiment(spark):
    """
    Load news data and sentiment scores, then merge them together.
    
    Args:
        spark (SparkSession): Active Spark session
        
    Returns:
        DataFrame: Merged news and sentiment data filtered for 2020-2023
    """
    # Load cleaned news data
    df_news = spark.read.parquet('s3a://fnf-bucket/silver/news_data_clean')
    print(f"News data count: {df_news.count()}")
    
    # Load sentiment analysis results
    df_sentiment = spark.read.parquet('s3a://fnf-bucket/silver/sentiment_data')
    print(f"Sentiment data count: {df_sentiment.count()}")
    
    # Merge news and sentiment data on article number
    merged_df = df_news.join(df_sentiment, on="No", how="inner")
    merged_df.show(1)
    
    # Display sentiment distribution
    merged_df.groupBy(F.col("predicted_sentiment")).count().show()
    
    # Filter data for the target date range (2020-2023)
    filtered_df = merged_df.filter(
        F.col("Date").between("2020-01-01", "2023-12-31")
    )
    
    print(f"Filtered data count: {filtered_df.count()}")
    filtered_df.show(5)
    filtered_df.groupBy(F.col("predicted_sentiment")).count().show()
    filtered_df.printSchema()
    
    return filtered_df


def aggregate_sentiment_by_stock_date(filtered_df):
    """
    Aggregate sentiment scores by stock symbol and date.
    
    Multiple news articles for the same stock on the same day are aggregated
    by averaging sentiment scores and counting article volume.
    
    Args:
        filtered_df (DataFrame): Filtered news-sentiment data
        
    Returns:
        DataFrame: Aggregated sentiment data by date and stock symbol
    """
    # Aggregate sentiment scores by date and stock symbol
    agg_df = filtered_df.groupBy("Date", "Stock_symbol").agg(
        F.avg("negative").alias("negative"),
        F.avg("neutral").alias("neutral"),
        F.avg("positive").alias("positive"),
        F.count("No").alias("article_count")
    )
    
    # Recalculate predicted sentiment based on highest average probability
    agg_df = agg_df.withColumn(
        "predicted_sentiment",
        F.when((F.col("positive") >= F.col("neutral")) & (F.col("positive") >= F.col("negative")), "positive")
         .when((F.col("neutral") >= F.col("positive")) & (F.col("neutral") >= F.col("negative")), "neutral")
         .otherwise("negative")
    )
    
    print(f"Aggregated sentiment data count: {agg_df.count()}")
    
    return agg_df


def save_aggregated_sentiment(agg_df):
    """
    Save aggregated sentiment data to S3.
    
    Args:
        agg_df (DataFrame): Aggregated sentiment data
    """
    agg_df.write \
        .mode('overwrite') \
        .option('compression', 'snappy') \
        .parquet('s3a://fnf-bucket/silver/sentiment_data_agg')
    
    print("Aggregated sentiment data saved successfully")


def merge_stock_prices_with_sentiment(spark):
    """
    Merge stock price data with aggregated sentiment data.
    
    Performs a left join to retain all stock price records, filling missing
    sentiment values with neutral scores.
    
    Args:
        spark (SparkSession): Active Spark session
        
    Returns:
        DataFrame: Stock prices merged with sentiment data
    """
    # Load stock price data
    df_stock = spark.read.parquet("s3a://fnf-bucket/silver/stock_price_data")
    
    # Load aggregated sentiment data
    df_sentiment = spark.read.parquet("s3a://fnf-bucket/silver/sentiment_data_agg")
    
    # Filter stock price data for target date range
    df_stock = df_stock.filter(
        F.col("Date").between("2020-01-01", "2023-12-31")
    )
    
    print("Stock price schema:")
    df_stock.printSchema()
    print("Sentiment schema:")
    df_sentiment.printSchema()
    
    # Perform left join to keep all stock price records
    joined_df = df_stock.join(
        df_sentiment,
        on=['Date', 'Stock_symbol'],
        how='left'
    )
    
    # Add flag indicating whether news exists for that day
    joined_df = joined_df.withColumn(
        'has_news',
        F.when(F.col('article_count').isNotNull(), True).otherwise(False)
    )
    
    # Fill null sentiment scores with neutral values
    # Assumption: No news = neutral sentiment
    joined_df = joined_df.fillna({
        'negative': 0.0,
        'positive': 0.0,
        'neutral': 1.0,
        'article_count': 0,
        'predicted_sentiment': 'neutral'
    })
    
    joined_df.printSchema()
    
    return joined_df


def save_merged_price_sentiment(joined_df):
    """
    Save merged price and sentiment data to S3.
    
    Args:
        joined_df (DataFrame): Merged stock price and sentiment data
    """
    joined_df.write \
        .mode('overwrite') \
        .option('compression', 'snappy') \
        .parquet('s3a://fnf-bucket/silver/news_price_with_sentiment')
    
    print("Merged price and sentiment data saved successfully")


def create_stock_categories_dataframe(spark):
    """
    Create a DataFrame mapping stock symbols to industry categories.
    
    Args:
        spark (SparkSession): Active Spark session
        
    Returns:
        DataFrame: Stock symbol to category mapping
    """
    # Define stock categories by industry sector
    stock_data = {
        "Technology": ["AAPL", "MSFT", "GOOG", "NVDA", "AMD", "INTC", "ORCL", "CRM", "ADBE", "QCOM", 
                       "IBM", "AVGO", "MU", "NOW", "PANW", "CRWD", "ANET", "FTNT", "TXN", "ADI", 
                       "KLAC", "LRCX", "AMAT", "SWKS", "ON", "DELL", "HPE", "HPQ", "CDNS", "ADSK", 
                       "WDAY", "DDOG", "AKAM", "NTAP", "SMCI", "EPAM", "VRSN", "GDDY", "FICO", "TYL", 
                       "BR", "FFIV", "FDS", "ZBRA", "TER", "MPWR", "GLW", "TRMB"],
        "Healthcare": ["ABT", "LLY", "REGN", "GILD", "BIIB", "VRTX", "ABBV", "AMGN", "MRK", "BSX", 
                       "DXCM", "ZTS", "MDT", "BAX", "EW", "ALGN", "HOLX", "PODD", "SYK", "TMO", 
                       "DHR", "RMD", "WST", "CI", "HUM", "HCA", "UHS", "DVA", "MCK", "CAH", 
                       "LH", "DGX", "CRL", "INCY"],
        "Financial_Services": ["V", "PYPL", "GS", "MS", "WFC", "C", "BX", "KKR", "APO", "COF", 
                               "USB", "PNC", "AIG", "BK", "KEY", "MTB", "FITB", "HBAN", "TFC", "RF", 
                               "CFG", "NTRS", "RJF", "IBKR", "AXP", "FIS", "GPN", "ICE", "CME", "NDAQ", 
                               "CBOE", "MCO", "MSCI", "AON", "AJG", "ACGL", "PGR", "ALL", "TRV", "AFL", 
                               "HIG", "PRU", "PFG", "CINF", "AIZ", "BRO", "ERIE"],
        "Consumer_Discretionary": ["AMZN", "TSLA", "NKE", "SBUX", "CMG", "DPZ", "TGT", "COST", "BBY", 
                                   "TJX", "ROST", "DLTR", "ULTA", "DRI", "MCD", "YUM", "BKNG", "EXPE", 
                                   "MGM", "RCL", "NCLH", "LYV", "ORLY", "AZO", "GM", "F", "DHI", "LEN", 
                                   "PHM", "NVR", "TPR", "RL", "DECK", "WSM", "CVNA", "DAL", "UAL", "LUV", "UBER"],
        "Consumer_Staples": ["WMT", "KO", "PEP", "PM", "GIS", "KHC", "CAG", "TSN", "CPB", "CL", 
                             "CLX", "KMB", "CHD", "MKC", "HRL", "TAP", "MNST", "EL", "SYY", "ADM"],
        "Energy": ["XOM", "CVX", "COP", "OXY", "SLB", "HAL", "DVN", "EOG", "FANG", "VLO", 
                   "PSX", "APA", "BKR", "EQT", "OKE", "KMI", "TRGP", "CTRA"],
        "Industrials": ["BA", "CAT", "GE", "DE", "UPS", "FDX", "NOC", "LMT", "UNP", "CSX", 
                        "NSC", "ODFL", "EMR", "ETN", "PCAR", "SWK", "CMI", "DOV", "ROK", "PH", 
                        "CARR", "OTIS", "LII", "ALLE", "APH", "WAB", "APTV", "PWR", "AME", "HUBB", 
                        "EME", "FTV", "TT", "LHX", "LDOS", "MSI", "GWW", "FAST", "SNA", "ROL", 
                        "CTAS", "URI", "CPRT", "CHRW", "VRSK", "ROP", "WAT", "A", "MTD", "XYL", 
                        "GNRC", "NDSN", "PNR", "AOS", "DOC", "TECH", "CRH"],
        "Materials": ["LIN", "APD", "NEM", "NUE", "DOW", "DD", "LYB", "CF", "MOS", "ALB", 
                      "VMC", "MLM", "SHW", "ECL", "CTVA", "BG", "PKG", "AVY", "WY"],
        "Real_Estate": ["AMT", "PLD", "PSA", "O", "VICI", "CCI", "EQR", "AVB", "VTR", "ARE", 
                        "WELL", "EXR", "ESS", "KIM", "FRT", "REG", "HST", "BXP", "CPT"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "PCG", "ED", "PEG", "EIX", 
                      "XEL", "WEC", "ES", "DTE", "AES", "AEE", "ETR", "CMS", "CNP", "ATO", 
                      "NRG", "PNW", "EVRG", "LNT", "AWK"],
        "Communication_Services": ["DIS", "CMCSA", "TMUS", "CHTR", "EA", "TTWO", "MTCH", "NWSA", 
                                   "NWS", "FOXA", "FOX"],
        "Miscellaneous": ["HAS", "TSCO", "GPC", "L", "LW", "COO", "GL", "FIX", "VST", "GEN", 
                          "COR", "ARES", "TPL", "CSGP", "POOL"]
    }
    
    # Flatten the dictionary into a list of (symbol, category) tuples
    flattened_data = []
    for category, symbols in stock_data.items():
        for symbol in symbols:
            flattened_data.append((symbol, category))
    
    # Define schema for categories DataFrame
    schema = StructType([
        StructField("Stock_symbol", StringType(), False),
        StructField("category", StringType(), False)
    ])
    
    # Create DataFrame from flattened data
    categories_df = spark.createDataFrame(flattened_data, schema)
    print(f"Categories data count: {categories_df.count()}")
    
    return categories_df


def add_categories_and_save(spark, categories_df):
    """
    Add category labels to stock data and save partitioned by category.
    
    Args:
        spark (SparkSession): Active Spark session
        categories_df (DataFrame): Stock symbol to category mapping
    """
    # Load merged price and sentiment data
    df = spark.read.parquet('s3a://fnf-bucket/silver/news_price_with_sentiment')
    
    # Join with categories, filling unmapped stocks with "Miscellaneous"
    merged_df = df.join(
        categories_df,
        on=['Stock_symbol'],
        how='left'
    ).fillna("Miscellaneous", subset=["category"])
    
    # Save data partitioned by industry category for efficient ML training
    merged_df.write \
        .mode('overwrite') \
        .option('compression', 'snappy') \
        .partitionBy('category') \
        .parquet('s3a://fnf-bucket/silver/categorical_stocks_data')
    
    print("Categorized stock data saved successfully")


def main():
    """
    Main execution function for sentiment data merging pipeline.
    """
    print("Starting sentiment data merge pipeline...")
    
    # Initialize Spark session
    spark = create_spark_session()
    
    try:
        # Step 1: Load and merge news with sentiment data
        print("\nStep 1: Loading and merging news with sentiment data...")
        filtered_df = load_and_merge_news_sentiment(spark)
        
        # Step 2: Aggregate sentiment by stock and date
        print("\nStep 2: Aggregating sentiment by stock and date...")
        agg_df = aggregate_sentiment_by_stock_date(filtered_df)
        
        # Step 3: Save aggregated sentiment data
        print("\nStep 3: Saving aggregated sentiment data...")
        save_aggregated_sentiment(agg_df)
        
        # Step 4: Merge stock prices with sentiment
        print("\nStep 4: Merging stock prices with sentiment data...")
        joined_df = merge_stock_prices_with_sentiment(spark)
        
        # Step 5: Save merged price and sentiment data
        print("\nStep 5: Saving merged price and sentiment data...")
        save_merged_price_sentiment(joined_df)
        
        # Step 6: Create stock categories mapping
        print("\nStep 6: Creating stock categories mapping...")
        categories_df = create_stock_categories_dataframe(spark)
        
        # Step 7: Add categories and save final dataset
        print("\nStep 7: Adding categories and saving final dataset...")
        add_categories_and_save(spark, categories_df)
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred during pipeline execution: {str(e)}")
        raise
    
    finally:
        # Stop Spark session
        spark.stop()
        print("Spark session stopped")


if __name__ == "__main__":
    main()
