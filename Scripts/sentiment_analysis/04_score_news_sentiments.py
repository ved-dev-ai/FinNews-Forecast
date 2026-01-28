"""
Sentiment Analysis Script for Financial News
=============================================

This script performs sentiment analysis on financial news summaries using FinBERT,
a pre-trained model specifically designed for financial text analysis.

The script:
1. Loads the FinBERT model and tokenizer
2. Reads parquet data containing news summaries
3. Performs batch sentiment analysis
4. Outputs sentiment scores (negative, neutral, positive) and predicted sentiment
5. Saves results to a parquet file

Author: FinNews-Forecast Team
Date: January 2026
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import argparse
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configuration
MODEL = "ProsusAI/finbert"
BATCH_SIZE = 64
SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def initialize_model(model_name=MODEL):
    """
    Initialize the FinBERT model and tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        
    Returns:
        tuple: (tokenizer, model) - Initialized tokenizer and model
    """
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model loaded on GPU")
    else:
        logger.info("Model loaded on CPU")
    
    return tokenizer, model


def process_batch(batch, tokenizer, model, labels=SENTIMENT_LABELS):
    """
    Process a batch of texts and return sentiment predictions.
    
    Args:
        batch (list): List of text strings to analyze
        tokenizer: The tokenizer for the model
        model: The sentiment analysis model
        labels (list): List of sentiment labels
        
    Returns:
        torch.Tensor: Probability distributions for each text in the batch
    """
    # Tokenize the batch
    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move inputs to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    
    return probs


def analyze_sentiment(input_path, output_path, batch_size=BATCH_SIZE):
    """
    Analyze sentiment of news summaries from parquet dataset.
    
    Args:
        input_path (str): Path to input parquet dataset
        output_path (str): Path to save output parquet file
        batch_size (int): Number of samples to process in each batch
        
    Returns:
        pd.DataFrame: DataFrame containing sentiment analysis results
    """
    logger.info(f"Starting sentiment analysis")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Batch size: {batch_size}")
    
    # Initialize model and tokenizer
    tokenizer, model = initialize_model()
    
    # Load the parquet dataset
    logger.info("Loading parquet dataset...")
    dataset = pq.ParquetDataset(input_path)
    
    results = []
    total_fragments = len(list(dataset.fragments))
    
    # Process each fragment in the dataset
    for fragment_idx, fragment in enumerate(dataset.fragments, 1):
        logger.info(f"Processing fragment {fragment_idx}/{total_fragments}")
        
        # Convert fragment to pandas DataFrame
        table = fragment.to_table()
        df = table.to_pandas()
        
        # Extract text summaries
        texts = df["Lsa_summary"].tolist()
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Fragment {fragment_idx} batches"):
            batch = texts[i:i + batch_size]
            
            # Get sentiment probabilities for the batch
            probs = process_batch(batch, tokenizer, model, SENTIMENT_LABELS)
            
            # Store results for each item in the batch
            for j in range(len(batch)):
                result = {
                    "No": df.iloc[i + j]["No"]
                }
                
                # Add probability scores for each sentiment label
                for label_idx, label_name in enumerate(SENTIMENT_LABELS):
                    result[label_name] = probs[j][label_idx].item()
                
                # Add the predicted sentiment (highest probability)
                result["predicted_sentiment"] = SENTIMENT_LABELS[torch.argmax(probs[j]).item()]
                
                results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    logger.info(f"Processed {len(results_df)} records")
    
    # Save results to parquet file
    logger.info(f"Saving results to {output_path}")
    results_df.to_parquet(output_path, index=False)
    logger.info("Sentiment analysis completed successfully")
    
    return results_df


def main():
    """
    Main function to run the sentiment analysis script.
    """
    parser = argparse.ArgumentParser(
        description='Perform sentiment analysis on financial news summaries using FinBERT'
    )
    parser.add_argument(
        '--input',
        type=str,
        default="../Datasets/data_for_finbert/",
        help='Path to input parquet dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="../Datasets/sentiment_data.parquet",
        help='Path to save output parquet file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size for processing'
    )
    
    args = parser.parse_args()
    
    # Run sentiment analysis
    try:
        analyze_sentiment(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
