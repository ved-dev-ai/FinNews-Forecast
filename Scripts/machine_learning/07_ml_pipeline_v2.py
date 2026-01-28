"""
Machine Learning Pipeline for Stock Price Prediction (ML_2)

This script performs feature engineering, preprocessing, and model training
using XGBoost to predict stock price movements based on technical indicators
and sentiment analysis.

The pipeline processes multiple stock categories, creates technical and sentiment-based
features, and trains XGBoost models using a rolling window approach for time-series validation.

Author: FinNews-Forecast Team
Date: 2026
"""

# Standard library imports
import os

# Third-party data manipulation libraries
import pandas as pd
import numpy as np

# Machine learning and model persistence
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    classification_report
)
from xgboost import XGBClassifier


def load_data(csv_path):
    """
    Load stock data from CSV file.
    
    Args:
        csv_path (str): Absolute or relative path to the CSV file containing stock data.
    
    Returns:
        pd.DataFrame: DataFrame containing stock price data, sentiment scores, and metadata.
    """
    return pd.read_csv(csv_path)


def create_labels(df):
    """
    Create binary labels based on 5-day forward returns using majority voting.
    
    The labeling strategy uses a forward-looking window to determine if the stock
    will experience mostly positive returns in the next 5 trading days.
    Label = 1 if >= 3 out of next 5 days are up days, else 0
    
    Args:
        df (pd.DataFrame): Input DataFrame with stock price data including Date,
                          Stock_symbol, and Close price columns.
    
    Returns:
        pd.DataFrame: Original DataFrame with added columns:
                     - Daily_Return: Percentage change in closing price
                     - Up_Day: Binary indicator (1=up, 0=down/flat)
                     - Up_Count_5d: Count of up days in next 5-day window
                     - Label: Binary target variable (1=bullish, 0=bearish)
    """
    # Sort data chronologically by stock symbol and date for proper time-series handling
    df = df.sort_values(['Stock_symbol', 'Date']).reset_index(drop=True)

    # Step 1: Compute daily returns (percentage change) for each stock independently
    df['Daily_Return'] = df.groupby('Stock_symbol')['Close'].pct_change()

    # Step 2: Create binary indicator for up days (positive returns)
    # 1 = price increased, 0 = price decreased or remained flat
    df['Up_Day'] = (df['Daily_Return'] > 0).astype(int)

    # Step 3: Count UP days in the next 5-day forward-looking window
    # shift(-1) moves the window forward, ensuring we're looking at future performance
    # min_periods=5 ensures we only calculate when a full 5-day window is available
    df['Up_Count_5d'] = df.groupby('Stock_symbol')['Up_Day'].transform(
        lambda x: x.shift(-1).rolling(window=5, min_periods=5).sum()
    )

    # Step 4: Apply majority voting rule to create binary classification label
    # Label=1 (bullish) if >= 3 out of 5 future days show positive returns
    # Label=0 (bearish) otherwise
    df['Label'] = np.where(df['Up_Count_5d'] >= 3, 1, 0)

    # Step 5: Remove rows without complete future window (typically the last 5 days)
    df_filtered = df.dropna(subset=['Up_Count_5d', 'Label'])

    # Display data quality metrics
    print(f"Original samples: {len(df)}")
    print(f"After filtering: {len(df_filtered)}")
    print("\nClass distribution:")
    print(df_filtered['Label'].value_counts())
    
    # Return original DataFrame with new columns (not filtered)
    return df


def engineer_return_features(df):
    """
    Create return-based momentum features.
    
    Calculates percentage price changes over different time horizons to capture
    short-term, medium-term, and longer-term momentum signals.
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data including Adj Close.
    
    Returns:
        pd.DataFrame: DataFrame with added return features:
                     - Return_1d: 1-day return (short-term momentum)
                     - Return_5d: 5-day return (weekly momentum)
                     - Return_20d: 20-day return (~1 month momentum)
    """
    # Group by stock symbol to ensure calculations are done per stock
    g = df.groupby("Stock_symbol")
    
    # Calculate returns over different time horizons
    df["Return_1d"]  = g["Adj Close"].pct_change(1)   # Daily momentum
    df["Return_5d"]  = g["Adj Close"].pct_change(5)   # Weekly momentum
    df["Return_20d"] = g["Adj Close"].pct_change(20)  # Monthly momentum
    
    return df


def engineer_moving_average_features(df):
    """
    Create moving average-based technical indicators.
    
    Generates features that capture price trends relative to moving averages,
    which are commonly used in technical analysis to identify support/resistance
    levels and trend direction.
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data including Adj Close.
    
    Returns:
        pd.DataFrame: DataFrame with added MA features:
                     - MA5: 5-day simple moving average
                     - MA20: 20-day simple moving average
                     - MA5_Ratio: Current price relative to MA5 (>1 = above MA)
                     - MA20_Ratio: Current price relative to MA20 (>1 = above MA)
                     - Trend_5_20: Relative position of MA5 to MA20 (trend strength)
    """
    # Group by stock symbol for independent calculations
    g = df.groupby("Stock_symbol")
    
    # Calculate simple moving averages
    df["MA5"]  = g["Adj Close"].transform(lambda x: x.rolling(5).mean())   # Short-term MA
    df["MA20"] = g["Adj Close"].transform(lambda x: x.rolling(20).mean())  # Long-term MA

    # Calculate price position relative to moving averages
    df["MA5_Ratio"]  = df["Adj Close"] / df["MA5"]   # Price vs short-term trend
    df["MA20_Ratio"] = df["Adj Close"] / df["MA20"]  # Price vs long-term trend
    
    # Calculate trend strength: positive when MA5 > MA20 (bullish crossover)
    df["Trend_5_20"] = (df["MA5"] - df["MA20"]) / df["MA20"]
    
    return df


def engineer_extrema_features(df):
    """
    Create features based on distance to 20-day price extremes.
    
    Measures how close the current price is to recent highs and lows,
    which can indicate overbought/oversold conditions or breakout potential.
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data including High, Low, Adj Close.
    
    Returns:
        pd.DataFrame: DataFrame with added extrema features:
                     - High_20: Highest price in last 20 days (resistance level)
                     - Low_20: Lowest price in last 20 days (support level)
                     - Dist_20High: Distance to 20-day high (negative = below high)
                     - Dist_20Low: Distance to 20-day low (positive = above low)
    """
    # Group by stock symbol
    g = df.groupby("Stock_symbol")
    
    # Find 20-day rolling extremes
    df["High_20"] = g["High"].transform(lambda x: x.rolling(20).max())  # Resistance level
    df["Low_20"]  = g["Low"].transform(lambda x: x.rolling(20).min())   # Support level

    # Calculate normalized distance to extremes (as percentage)
    df["Dist_20High"] = (df["Adj Close"] - df["High_20"]) / df["High_20"]  # Usually negative
    df["Dist_20Low"]  = (df["Adj Close"] - df["Low_20"]) / df["Low_20"]    # Usually positive
    
    return df


def engineer_volatility_features(df):
    """
    Create volatility-based risk indicators.
    
    Volatility measures help quantify market risk and can signal regime changes.
    Higher volatility often precedes significant price movements.
    
    Args:
        df (pd.DataFrame): DataFrame with stock price data including Adj Close, Open.
                          Must have Return_1d already calculated.
    
    Returns:
        pd.DataFrame: DataFrame with added volatility features:
                     - Vol_5d: 5-day rolling volatility (short-term risk)
                     - Vol_20d: 20-day rolling volatility (baseline risk)
                     - Vol_Ratio: Recent vs baseline volatility (>1 = elevated risk)
                     - Gap: Overnight gap between previous close and current open
    """
    # Group by stock symbol
    g = df.groupby("Stock_symbol")
    
    # Calculate rolling standard deviation of returns (volatility)
    df["Vol_5d"]  = g["Return_1d"].transform(lambda x: x.rolling(5).std())   # Short-term vol
    df["Vol_20d"] = g["Return_1d"].transform(lambda x: x.rolling(20).std())  # Long-term vol
    df["Vol_Ratio"] = df["Vol_5d"] / df["Vol_20d"]  # Volatility regime indicator
    
    # Calculate overnight gap (market sentiment overnight)
    prev_close = g["Adj Close"].shift(1)
    df["Gap"] = (df["Open"] - prev_close) / prev_close  # Positive = gap up, Negative = gap down
    
    return df


def engineer_volume_features(df):
    """
    Create trading volume-based features.
    
    Volume analysis helps confirm price movements and identify institutional activity.
    High volume on price moves suggests strong conviction.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data including Volume.
                          Must have Return_1d already calculated.
    
    Returns:
        pd.DataFrame: DataFrame with added volume features:
                     - Vol_Change: Day-over-day volume change
                     - Vol_MA20: 20-day average volume (baseline activity)
                     - Vol_Ratio_20: Current volume vs baseline (>1 = high activity)
                     - PV_Score: Price-Volume score (strong moves with high volume)
                     - Vol_MA5: 5-day average volume (recent trend)
                     - Vol_Trend: Short vs long-term volume trend
    """
    # Group by stock symbol
    g = df.groupby("Stock_symbol")
    
    # Calculate volume changes
    df["Vol_Change"] = g["Volume"].pct_change()  # Day-to-day volume change
    
    # Calculate volume moving averages
    df["Vol_MA20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())  # Baseline volume
    df["Vol_Ratio_20"] = df["Volume"] / df["Vol_MA20"]  # Relative volume indicator
    
    # Create Price-Volume score: combines return direction with volume confirmation
    df["PV_Score"] = df["Return_1d"] * df["Vol_Ratio_20"]  # Higher = stronger conviction
    
    # Calculate volume trend
    df["Vol_MA5"] = g["Volume"].transform(lambda x: x.rolling(5).mean())  # Recent volume
    df["Vol_Trend"] = df["Vol_MA5"] / df["Vol_MA20"]  # Increasing trend when >1
    
    return df


def engineer_sentiment_features(df):
    """
    Create news sentiment-based features.
    
    Leverages FinBERT sentiment scores from financial news articles to capture
    market sentiment and narrative around each stock.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment scores including positive and negative.
    
    Returns:
        pd.DataFrame: DataFrame with added sentiment features:
                     - Sent_Net: Net sentiment (positive - negative)
                     - Sent_MA5: 5-day moving average of sentiment (trend)
                     - Sent_Mom5: 5-day sentiment momentum (change in sentiment)
    """
    # Group by stock symbol
    g = df.groupby("Stock_symbol")
    
    # Calculate net sentiment (range: -1 to +1)
    df["Sent_Net"] = df["positive"] - df["negative"]  # Positive = bullish news
    
    # Calculate sentiment trend (smoothed sentiment)
    df["Sent_MA5"] = g["Sent_Net"].transform(
        lambda x: x.rolling(5).mean()
    )
    
    # Calculate sentiment momentum (change in sentiment over 5 days)
    df["Sent_Mom5"] = g["Sent_Net"].diff(5)  # Positive = improving sentiment
    
    return df


def engineer_news_features(df):
    """
    Create news intensity features.
    
    News volume can indicate increased attention and potential volatility.
    Unusual news activity often precedes significant price movements.
    
    Args:
        df (pd.DataFrame): DataFrame with article_count column.
    
    Returns:
        pd.DataFrame: DataFrame with added news features:
                     - Art_MA20: 20-day average article count (baseline coverage)
                     - News_Intensity: Current vs baseline news coverage (>1 = high attention)
    """
    # Group by stock symbol
    g = df.groupby("Stock_symbol")
    
    # Calculate baseline news coverage
    df["Art_MA20"] = g["article_count"].transform(
        lambda x: x.rolling(20).mean()
    )

    # Calculate relative news intensity (>1 indicates unusual news volume)
    df["News_Intensity"] = df["article_count"] / df["Art_MA20"]
    
    return df


def engineer_lag_features(df):
    """
    Create lagged features to capture temporal dependencies.
    
    Lagged features allow the model to learn from historical patterns while
    avoiding data leakage (using only information available at prediction time).
    
    Args:
        df (pd.DataFrame): DataFrame with previously calculated features.
                          Must have Return_1d, Return_5d, and Sent_Net columns.
    
    Returns:
        pd.DataFrame: DataFrame with added lag features:
                     - Ret_Lag1: Previous day's return (short-term reversal/momentum)
                     - Ret_Lag5: Return from 5 days ago (weekly patterns)
                     - Sent_Lag3: Sentiment from 3 days ago (sentiment lead time)
    """
    # Group by stock symbol to ensure lags are calculated per stock
    g = df.groupby("Stock_symbol")
    
    # Create lagged return features
    df["Ret_Lag1"]  = g["Return_1d"].shift(1)   # T-1 return
    df["Ret_Lag5"]  = g["Return_5d"].shift(5)   # T-5 return
    
    # Create lagged sentiment feature (sentiment may have delayed impact)
    df["Sent_Lag3"] = g["Sent_Net"].shift(3)    # T-3 sentiment
    
    return df


def engineer_all_features(df):
    """
    Apply all feature engineering steps in proper sequence.
    
    This is the master function that orchestrates the entire feature engineering
    pipeline. Order matters: some features depend on features created earlier.
    
    Args:
        df (pd.DataFrame): Raw stock data with OHLCV prices, sentiment scores,
                          and article counts.
    
    Returns:
        pd.DataFrame: Fully engineered DataFrame with all technical indicators,
                     sentiment features, and target labels.
    """
    # Sort data chronologically by stock and date (critical for time-series features)
    df = df.sort_values(["Stock_symbol", "Date"]).reset_index(drop=True)
    
    # Apply feature engineering in dependency order
    df = create_labels(df)                      # Create target variable
    df = engineer_return_features(df)           # Return-based features (needed for other features)
    df = engineer_moving_average_features(df)   # Moving average indicators
    df = engineer_extrema_features(df)          # High/low distance features
    df = engineer_volatility_features(df)       # Volatility indicators (needs Return_1d)
    df = engineer_volume_features(df)           # Volume features (needs Return_1d)
    df = engineer_sentiment_features(df)        # Sentiment features
    df = engineer_news_features(df)             # News intensity features
    df = engineer_lag_features(df)              # Lagged features (must be last)
    
    return df


def normalize_features(df, features, window=252):
    """
    Apply z-score normalization using rolling window.
    
    Normalizes features using historical statistics to ensure stationarity and
    prevent look-ahead bias. Uses a rolling window to adapt to changing market regimes.
    
    Args:
        df (pd.DataFrame): DataFrame with features to normalize.
        features (list): List of feature column names to normalize.
        window (int): Rolling window size in days (default: 252 ~= 1 trading year).
    
    Returns:
        pd.DataFrame: DataFrame with added normalized features (suffix: "_z").
                     Normalized features have approximately mean=0, std=1.
    """
    # Group by stock symbol to normalize each stock independently
    g = df.groupby("Stock_symbol")
    
    # Normalize each feature using rolling z-score
    for f in features:
        # Calculate rolling mean (center of distribution)
        rolling_mean = g[f].transform(
            lambda x: x.rolling(window).mean()
        )

        # Calculate rolling standard deviation (spread of distribution)
        rolling_std = g[f].transform(
            lambda x: x.rolling(window).std()
        )

        # Apply z-score normalization: (value - mean) / std
        # Add small epsilon (1e-8) to prevent division by zero
        df[f + "_z"] = (df[f] - rolling_mean) / (rolling_std + 1e-8)
    
    return df


def make_pipeline():
    """
    Create XGBoost classification pipeline with optimized hyperparameters.
    
    XGBoost is chosen for its strong performance on tabular data, ability to handle
    non-linear relationships, and built-in regularization to prevent overfitting.
    
    Hyperparameters are tuned for financial time-series prediction:
    - Conservative learning rate and tree depth to reduce overfitting
    - Subsampling to improve generalization
    - AUC metric for imbalanced classification
    
    Returns:
        sklearn.pipeline.Pipeline: Configured XGBoost classification pipeline.
    """
    pipe = Pipeline([
        ("model", XGBClassifier(
            n_estimators=400,        # Number of boosting rounds
            max_depth=5,             # Maximum tree depth (prevents overfitting)
            learning_rate=0.03,      # Conservative learning rate for stability
            subsample=0.8,           # Row sampling ratio (reduces overfitting)
            colsample_bytree=0.8,    # Column sampling ratio (feature randomization)
            eval_metric="auc",       # Optimize for AUC-ROC (good for imbalanced data)
            random_state=42,         # Reproducibility
            n_jobs=-1                # Use all CPU cores
        ))
    ])

    return pipe


def train_test_split(df, features_list, target, train_days=252, test_days=21, save_models=True, models_dir="../models"):
    """
    Perform rolling window train-test split.
    
    Args:
        df: DataFrame with data
        features_list: List of feature column names
        target: Target column name
        train_days: Number of days for training window (~1 year)
        test_days: Number of days for test window (~1 month)
        save_models: Whether to save trained models (default: True)
        models_dir: Directory to save models (default: "../models")
    
    Returns:
        List of results dictionaries
    """
    # Initialize results storage
    results = []
    
    # Create models directory if it doesn't exist and saving is enabled
    if save_models and not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    
    # Prepare data: ensure Date column is datetime type
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    dates = df["Date"].sort_values().unique()

    # Rolling window approach: train on past year, test on next month
    # Iterate with step size = test_days to avoid overlapping test periods
    for i in range(train_days, len(dates), test_days):
        
        # Define training window (past year)
        train_start = dates[i - train_days]
        train_end   = dates[i - 1]

        # Define test window (next month)
        test_start  = dates[i]
        test_end    = dates[min(i + test_days - 1, len(dates) - 1)]

        # Extract training and test data based on date windows
        train = df[
            (df["Date"] >= train_start) &
            (df["Date"] <= train_end)
        ]

        test = df[
            (df["Date"] >= test_start) &
            (df["Date"] <= test_end)
        ]

        # Skip if test set is empty (end of data)
        if len(test) == 0:
            continue

        # Separate features and target for training
        X_train = train[features_list]
        y_train = train[target]

        # Separate features and target for testing
        X_test  = test[features_list]
        y_test  = test[target]

        # Train the XGBoost model on training data
        pipe = make_pipeline()
        pipe.fit(X_train, y_train)

        # Generate predictions on test set
        y_prob = pipe.predict_proba(X_test)[:, 1]  # Probability of class 1 (bullish)
        y_pred = (y_prob > 0.5).astype(int)         # Binary predictions using 0.5 threshold

        # Calculate comprehensive evaluation metrics
        auc  = roc_auc_score(y_test, y_prob)                        # AUC-ROC (discrimination ability)
        acc  = accuracy_score(y_test, y_pred)                       # Overall accuracy
        precision = precision_score(y_test, y_pred, zero_division=0)  # Precision (positive predictive value)
        recall = recall_score(y_test, y_pred, zero_division=0)        # Recall (sensitivity)
        f1 = f1_score(y_test, y_pred, zero_division=0)                # F1 (harmonic mean of precision/recall)
        ll = log_loss(y_test, y_prob)                                 # Log loss (calibration quality)
        
        # Persist model with metadata for future inference
        # Persist trained model to disk with comprehensive metadata
        if save_models:
            # Create descriptive filename with training and test date ranges
            model_filename = f"model_train_{train_start.strftime('%Y%m%d')}_to_{train_end.strftime('%Y%m%d')}_test_{test_start.strftime('%Y%m%d')}_to_{test_end.strftime('%Y%m%d')}.pkl"
            model_path = os.path.join(models_dir, model_filename)
            
            # Package model with metadata for reproducibility and auditing
            model_data = {
                'model': pipe,                      # Trained pipeline
                'features': features_list,          # Feature names (for inference)
                'train_start': train_start,         # Training window start
                'train_end': train_end,             # Training window end
                'test_start': test_start,           # Test window start
                'test_end': test_end,               # Test window end
                'metrics': {                        # Performance metrics
                    'auc': auc,
                    'accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'log_loss': ll
                },
                'n_train': len(train),              # Training samples count
                'n_test': len(test)                 # Test samples count
            }
            
            # Serialize model and metadata using joblib
            joblib.dump(model_data, model_path)

        # Store results for this time window
        results.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "auc": auc,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "log_loss": ll,
            "n_train": len(train),
            "n_test": len(test),
            "model_path": model_path if save_models else None
        })

        # Print performance metrics for this test window
        print(
            f"{test_start} | "
            f"AUC: {auc:.3f} | "
            f"ACC: {acc:.3f} | "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f} | "
            f"Log Loss: {ll:.3f}"
        )

    # Return all results across all time windows
    return results


def print_results(results):
    """
    Print overall performance metrics and year-by-year breakdown.
    
    Aggregates results across all rolling windows to provide summary statistics
    and identify temporal performance patterns.
    
    Args:
        results (list): List of result dictionaries from train_test_split.
    
    Returns:
        pd.DataFrame: DataFrame containing all results for further analysis.
    """
    # Convert results list to DataFrame for easier aggregation
    results_df = pd.DataFrame(results)

    # Print aggregate performance metrics
    print("\n==== OVERALL PERFORMANCE ====\n")

    print("Mean AUC: ", results_df["auc"].mean())
    print("Mean ACC: ", results_df["accuracy"].mean())
    print("Mean Precision: ", results_df["precision"].mean())
    print("Mean Recall: ", results_df["recall"].mean())
    print("Mean F1: ", results_df["f1"].mean())
    print("Mean Log Loss: ", results_df["log_loss"].mean())

    # Print year-by-year performance (identifies regime-specific performance)
    print("\nBy Year:")

    results_df["year"] = results_df["test_start"].dt.year

    print(
        results_df.groupby("year")[["auc", "accuracy"]].mean()
    )
    
    return results_df


def main():
    """
    Main execution function - orchestrates the entire ML pipeline.
    
    Processes all stock categories sequentially:
    1. Loads category-specific stock data
    2. Engineers technical and sentiment features
    3. Trains XGBoost models using rolling window validation
    4. Saves trained models with metadata
    5. Prints performance metrics
    
    The pipeline is designed to be robust to errors, continuing with remaining
    categories if one fails.
    """
    
    # Configuration: Define paths for data and model storage
    csv_dir = "../Datasets/category_csvs"        # Input data directory
    base_models_dir = "../models"                # Output models directory
    
    # Discover all category CSV files to process
    csv_files = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")])
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    # Process each stock category independently
    for csv_file in csv_files:
        # Construct full path to category CSV file
        csv_path = os.path.join(csv_dir, csv_file)
        
        # Extract category name from filename (remove "_stocks_data.csv" suffix)
        category_name = csv_file.replace("_stocks_data.csv", "")
        
        # Create category-specific subdirectory for models
        models_dir = os.path.join(base_models_dir, category_name)
        
        # Print category header for visibility
        print(f"\n{'='*70}")
        print(f"Processing: {category_name}")
        print(f"{'='*70}\n")
        
        try:
            # Step 1: Load raw data from CSV
            print(f"Loading data from {csv_file}...")
            df = load_data(csv_path)
            
            print(f"Data shape: {df.shape}")
            
            # Step 2: Apply feature engineering pipeline
            print("Engineering features...")
            df = engineer_all_features(df)
            
            # Step 3: Define feature set for modeling
            # Features are organized by category for interpretability
            features = [
                # Momentum features (price trends and moving average relationships)
                "Return_1d", "Return_5d", "Return_20d",
                "MA5_Ratio", "MA20_Ratio", "Trend_5_20",
                "Dist_20High", "Dist_20Low",

                # Volatility features (risk and market regime indicators)
                "Vol_5d", "Vol_20d",
                "Vol_Ratio", "Gap",

                # Volume features (trading activity and conviction)
                "Vol_Change", "Vol_Ratio_20", "PV_Score", "Vol_Trend",

                # Sentiment features (news-based market sentiment)
                "Sent_Net", "Sent_MA5", "Sent_Mom5",
                "News_Intensity",

                # Lag features (temporal dependencies)
                "Ret_Lag1", "Ret_Lag5", "Sent_Lag3"
            ]
            
            # Step 4: Apply rolling z-score normalization for stationarity
            print("Normalizing features...")
            df = normalize_features(df, features, window=252)
            
            # Step 5: Prepare final dataset for modeling
            # Use normalized features (suffix "_z") as model inputs
            final_features = [f + "_z" for f in features]
            # Remove rows with missing values in features or target
            df_model = df.dropna(subset=final_features + ["Label"])
            
            print(f"Total samples for modeling: {len(df_model)}")
            
            # Validate that we have sufficient data for training
            if len(df_model) == 0:
                print(f"WARNING: No valid samples for {category_name}. Skipping...\n")
                continue
            
            # Step 6: Train models using rolling window cross-validation
            # - Train window: 252 days (~1 trading year)
            # - Test window: 21 days (~1 trading month)
            # - Models are saved with metadata for future deployment
            print(f"\nTraining models with rolling window validation for {category_name}...\n")
            results = train_test_split(df_model, final_features, "Label", 
                                       train_days=252, test_days=21,
                                       save_models=True, models_dir=models_dir)
            
            # Step 7: Print aggregated performance metrics
            results_df = print_results(results)
            
            print(f"\nModels saved to: {models_dir}")
            
        except Exception as e:
            # Catch and log errors without stopping the entire pipeline
            print(f"ERROR processing {category_name}: {str(e)}")
            print(f"Skipping to next category...\n")
            continue
    
    # Print completion message
    print(f"\n{'='*70}")
    print("All categories processed!")
    print(f"{'='*70}")
    


# Entry point: execute main function when script is run directly
if __name__ == "__main__":
    main()
