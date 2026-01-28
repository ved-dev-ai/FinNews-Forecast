"""
Streamlit App for Stock Price Prediction ML Pipeline

Interactive interface to run backtests on different stock datasets
and view the results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import sys
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
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Set page config
st.set_page_config(
    page_title="Stock Price Prediction Backtest",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction ML Pipeline")
st.markdown("Run backtests on different stock categories and analyze the results")


def load_data(csv_path):
    """Load stock data from CSV file."""
    return pd.read_csv(csv_path)


def create_labels(df):
    """
    Create binary labels based on 5-day forward returns.
    
    Label = 1 if >= 3 out of next 5 days are up days, else 0
    """
    # Sort properly
    df = df.sort_values(['Stock_symbol', 'Date']).reset_index(drop=True)

    # Step 1: Compute daily returns per stock
    df['Daily_Return'] = df.groupby('Stock_symbol')['Close'].pct_change()

    # Step 2: Mark up/down days (1 = up, 0 = down or flat)
    df['Up_Day'] = (df['Daily_Return'] > 0).astype(int)

    # Step 3: Count UP days in next 5 days
    df['Up_Count_5d'] = df.groupby('Stock_symbol')['Up_Day'].transform(
        lambda x: x.shift(-1).rolling(window=5, min_periods=5).sum()
    )

    # Step 4: Majority vote (>=3 up days = UP)
    df['Label'] = np.where(df['Up_Count_5d'] >= 3, 1, 0)

    # Step 5: Remove rows without full future window
    df_filtered = df.dropna(subset=['Up_Count_5d', 'Label'])

    return df, df_filtered


def engineer_return_features(df):
    """Create return-based features."""
    g = df.groupby("Stock_symbol")
    
    df["Return_1d"]  = g["Adj Close"].pct_change(1)
    df["Return_5d"]  = g["Adj Close"].pct_change(5)
    df["Return_20d"] = g["Adj Close"].pct_change(20)
    
    return df


def engineer_moving_average_features(df):
    """Create moving average-based features."""
    g = df.groupby("Stock_symbol")
    
    df["MA5"]  = g["Adj Close"].transform(lambda x: x.rolling(5).mean())
    df["MA20"] = g["Adj Close"].transform(lambda x: x.rolling(20).mean())

    df["MA5_Ratio"]  = df["Adj Close"] / df["MA5"]
    df["MA20_Ratio"] = df["Adj Close"] / df["MA20"]
    
    df["Trend_5_20"] = (df["MA5"] - df["MA20"]) / df["MA20"]
    
    return df


def engineer_extrema_features(df):
    """Create 20-day high/low distance features."""
    g = df.groupby("Stock_symbol")
    
    df["High_20"] = g["High"].transform(lambda x: x.rolling(20).max())
    df["Low_20"]  = g["Low"].transform(lambda x: x.rolling(20).min())

    df["Dist_20High"] = (df["Adj Close"] - df["High_20"]) / df["High_20"]
    df["Dist_20Low"]  = (df["Adj Close"] - df["Low_20"]) / df["Low_20"]
    
    return df


def engineer_volatility_features(df):
    """Create volatility-based features."""
    g = df.groupby("Stock_symbol")
    
    df["Vol_5d"]  = g["Return_1d"].transform(lambda x: x.rolling(5).std())
    df["Vol_20d"] = g["Return_1d"].transform(lambda x: x.rolling(20).std())
    df["Vol_Ratio"] = df["Vol_5d"] / df["Vol_20d"]
    
    prev_close = g["Adj Close"].shift(1)
    df["Gap"] = (df["Open"] - prev_close) / prev_close
    
    return df


def engineer_volume_features(df):
    """Create volume-based features."""
    g = df.groupby("Stock_symbol")
    
    df["Vol_Change"] = g["Volume"].pct_change()
    
    df["Vol_MA20"] = g["Volume"].transform(lambda x: x.rolling(20).mean())
    df["Vol_Ratio_20"] = df["Volume"] / df["Vol_MA20"]
    
    df["PV_Score"] = df["Return_1d"] * df["Vol_Ratio_20"]
    
    df["Vol_MA5"] = g["Volume"].transform(lambda x: x.rolling(5).mean())
    df["Vol_Trend"] = df["Vol_MA5"] / df["Vol_MA20"]
    
    return df


def engineer_sentiment_features(df):
    """Create sentiment-based features."""
    g = df.groupby("Stock_symbol")
    
    df["Sent_Net"] = df["positive"] - df["negative"]
    
    df["Sent_MA5"] = g["Sent_Net"].transform(
        lambda x: x.rolling(5).mean()
    )
    
    df["Sent_Mom5"] = g["Sent_Net"].diff(5)
    
    return df


def engineer_news_features(df):
    """Create news intensity features."""
    g = df.groupby("Stock_symbol")
    
    df["Art_MA20"] = g["article_count"].transform(
        lambda x: x.rolling(20).mean()
    )

    df["News_Intensity"] = df["article_count"] / df["Art_MA20"]
    
    return df


def engineer_lag_features(df):
    """Create lagged features."""
    g = df.groupby("Stock_symbol")
    
    df["Ret_Lag1"]  = g["Return_1d"].shift(1)
    df["Ret_Lag5"]  = g["Return_5d"].shift(5)
    df["Sent_Lag3"] = g["Sent_Net"].shift(3)
    
    return df


def engineer_all_features(df):
    """Apply all feature engineering steps."""
    df = df.sort_values(["Stock_symbol", "Date"]).reset_index(drop=True)
    
    df, _ = create_labels(df)
    df = engineer_return_features(df)
    df = engineer_moving_average_features(df)
    df = engineer_extrema_features(df)
    df = engineer_volatility_features(df)
    df = engineer_volume_features(df)
    df = engineer_sentiment_features(df)
    df = engineer_news_features(df)
    df = engineer_lag_features(df)
    
    return df


def normalize_features(df, features, window=252):
    """Apply z-score normalization using rolling window."""
    g = df.groupby("Stock_symbol")
    
    for f in features:
        rolling_mean = g[f].transform(
            lambda x: x.rolling(window).mean()
        )

        rolling_std = g[f].transform(
            lambda x: x.rolling(window).std()
        )

        df[f + "_z"] = (df[f] - rolling_mean) / (rolling_std + 1e-8)
    
    return df


def make_pipeline():
    """Create XGBoost classification pipeline."""
    pipe = Pipeline([
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ))
    ])

    return pipe


def train_test_split(df, features_list, target, train_days=252, test_days=21):
    """
    Perform rolling window train-test split.
    
    Args:
        df: DataFrame with data
        features_list: List of feature column names
        target: Target column name
        train_days: Number of days for training window (~1 year)
        test_days: Number of days for test window (~1 month)
    
    Returns:
        Tuple of (List of results dictionaries, all_y_test, all_y_pred, predictions_df)
    """
    results = []
    all_y_test = []
    all_y_pred = []
    predictions_list = []
    
    df["Date"] = pd.to_datetime(df["Date"])
    dates = df["Date"].sort_values().unique()

    for i in range(train_days, len(dates), test_days):
        
        train_start = dates[i - train_days]
        train_end   = dates[i - 1]

        test_start  = dates[i]
        test_end    = dates[min(i + test_days - 1, len(dates) - 1)]

        # Split data
        train = df[
            (df["Date"] >= train_start) &
            (df["Date"] <= train_end)
        ]

        test = df[
            (df["Date"] >= test_start) &
            (df["Date"] <= test_end)
        ]

        if len(test) == 0:
            continue

        X_train = train[features_list]
        y_train = train[target]

        X_test  = test[features_list]
        y_test  = test[target]

        # Train model
        pipe = make_pipeline()
        pipe.fit(X_train, y_train)

        # Predict
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        # Evaluate
        auc  = roc_auc_score(y_test, y_prob)
        acc  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        ll = log_loss(y_test, y_prob)

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
            "n_test": len(test)
        })
        
        # Store predictions for confusion matrix
        all_y_test.extend(y_test.values)
        all_y_pred.extend(y_pred)
        
        # Store detailed predictions with metadata
        test_with_pred = test[[target, 'Date', 'Stock_symbol']].copy()
        test_with_pred['y_pred'] = y_pred
        test_with_pred['y_test'] = y_test.values
        predictions_list.append(test_with_pred)

    predictions_df = pd.concat(predictions_list, ignore_index=True)
    return results, np.array(all_y_test), np.array(all_y_pred), predictions_df


# Sidebar configuration
st.sidebar.header("Configuration")

# Get available CSV files
datasets_path = Path("./Datasets/category_csvs")
csv_files = list(datasets_path.glob("*.csv"))
csv_files_dict = {f.stem: str(f) for f in csv_files}

if not csv_files_dict:
    st.error("No CSV files found in ./Datasets/category_csvs")
    st.stop()

# Dropdown for file selection
selected_category = st.sidebar.selectbox(
    "Select Stock Category",
    options=list(csv_files_dict.keys()),
    index=0
)

# Display selected file info
st.sidebar.info(f"Selected: {selected_category}")

# Run backtest button
if st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True):
    
    with st.spinner("Running backtest... This may take a few minutes."):
        
        # Create a container for status messages
        status_container = st.container()
        
        with status_container:
            st.subheader("Processing Log")
            log_box = st.empty()
            log_messages = []
            
            # Redirect stdout to capture print statements
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Load data
                log_messages.append("📂 Loading data...")
                log_box.text_area("Log Output", value="\n".join(log_messages), height=200, disabled=True)
                
                csv_path = csv_files_dict[selected_category]
                df = load_data(csv_path)
                
                # Feature engineering
                log_messages.append(f"🔧 Engineering features for {len(df)} samples...")
                log_box.text_area("Log Output", value="\n".join(log_messages), height=200, disabled=True)
                
                df = engineer_all_features(df)
                
                # Define base features
                features = [
                    # Momentum
                    "Return_1d", "Return_5d", "Return_20d",
                    "MA5_Ratio", "MA20_Ratio", "Trend_5_20",
                    "Dist_20High", "Dist_20Low",

                    # Volatility
                    "Vol_5d", "Vol_20d",
                    "Vol_Ratio", "Gap",

                    # Volume
                    "Vol_Change", "Vol_Ratio_20", "PV_Score", "Vol_Trend",

                    # Sentiment
                    "Sent_Net", "Sent_MA5", "Sent_Mom5",
                    "News_Intensity",

                    # Lags
                    "Ret_Lag1", "Ret_Lag5", "Sent_Lag3"
                ]
                
                # Normalize features
                log_messages.append("📊 Normalizing features...")
                log_box.text_area("Log Output", value="\n".join(log_messages), height=200, disabled=True)
                
                df = normalize_features(df, features, window=252)
                
                # Prepare data for modeling
                final_features = [f + "_z" for f in features]
                df_model = df.dropna(subset=final_features + ["Label"])
                
                log_messages.append(f"✅ Total samples for modeling: {len(df_model)}")
                log_box.text_area("Log Output", value="\n".join(log_messages), height=200, disabled=True)
                
                # Model training and evaluation
                log_messages.append("\n🤖 Training models with rolling window validation...")
                log_box.text_area("Log Output", value="\n".join(log_messages), height=200, disabled=True)
                
                results, all_y_test, all_y_pred, predictions_df = train_test_split(df_model, final_features, "Label", 
                                          train_days=252, test_days=21)
                
                for i, result in enumerate(results, 1):
                    log_messages.append(
                        f"{result['test_start'].strftime('%Y-%m-%d')} | "
                        f"AUC: {result['auc']:.3f} | "
                        f"ACC: {result['accuracy']:.3f} | "
                        f"Precision: {result['precision']:.3f} | "
                        f"Recall: {result['recall']:.3f} | "
                        f"F1: {result['f1']:.3f} | "
                        f"Log Loss: {result['log_loss']:.3f}"
                    )
                    if i % 5 == 0:
                        log_box.text_area("Log Output", value="\n".join(log_messages), height=200, disabled=True)
                
                log_box.text_area("Log Output", value="\n".join(log_messages), height=200, disabled=True)
                
            finally:
                sys.stdout = old_stdout
        
        # Display results
        st.success("✅ Backtest completed successfully!")
        
        st.subheader("📊 Results Summary")
        
        results_df = pd.DataFrame(results)
        
        # Overall metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Mean AUC", f"{results_df['auc'].mean():.4f}")
        with col2:
            st.metric("Mean Accuracy", f"{results_df['accuracy'].mean():.4f}")
        with col3:
            st.metric("Mean Precision", f"{results_df['precision'].mean():.4f}")
        with col4:
            st.metric("Mean Recall", f"{results_df['recall'].mean():.4f}")
        with col5:
            st.metric("Mean F1", f"{results_df['f1'].mean():.4f}")
        with col6:
            st.metric("Mean Log Loss", f"{results_df['log_loss'].mean():.4f}")
        
        # Summary statistics table
        st.subheader("📋 Summary Statistics Table")
        summary_stats = pd.DataFrame({
            'Metric': ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Log Loss'],
            'Mean': [
                results_df['auc'].mean(),
                results_df['accuracy'].mean(),
                results_df['precision'].mean(),
                results_df['recall'].mean(),
                results_df['f1'].mean(),
                results_df['log_loss'].mean()
            ],
            'Std': [
                results_df['auc'].std(),
                results_df['accuracy'].std(),
                results_df['precision'].std(),
                results_df['recall'].std(),
                results_df['f1'].std(),
                results_df['log_loss'].std()
            ],
            'Min': [
                results_df['auc'].min(),
                results_df['accuracy'].min(),
                results_df['precision'].min(),
                results_df['recall'].min(),
                results_df['f1'].min(),
                results_df['log_loss'].min()
            ],
            'Max': [
                results_df['auc'].max(),
                results_df['accuracy'].max(),
                results_df['precision'].max(),
                results_df['recall'].max(),
                results_df['f1'].max(),
                results_df['log_loss'].max()
            ]
        })
        st.dataframe(summary_stats.style.format({
            'Mean': '{:.4f}',
            'Std': '{:.4f}',
            'Min': '{:.4f}',
            'Max': '{:.4f}'
        }), use_container_width=True)
        
        # Confusion Matrix
        st.subheader("🎯 Overall Confusion Matrix")
        cm = confusion_matrix(all_y_test, all_y_pred)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Create confusion matrix heatmap
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Down', 'Predicted Up'],
                y=['Actual Down', 'Actual Up'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                hovertemplate='%{y}<br>%{x}<br>Count: %{z}<extra></extra>'
            ))
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                height=400
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Confusion matrix metrics
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            
            cm_metrics = pd.DataFrame({
                'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives'],
                'Count': [tp, tn, fp, fn],
                'Percentage': [
                    f"{(tp/total)*100:.2f}%",
                    f"{(tn/total)*100:.2f}%",
                    f"{(fp/total)*100:.2f}%",
                    f"{(fn/total)*100:.2f}%",
                ]
            })
            st.dataframe(cm_metrics, use_container_width=True, height=180)
            
            # Additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            st.metric("Specificity (True Negative Rate)", f"{specificity:.4f}")
            st.metric("Sensitivity (True Positive Rate)", f"{sensitivity:.4f}")
        
        # Detailed results table
        st.subheader("📊 Detailed Results by Period")
        
        display_df = results_df.copy()
        display_df['test_start'] = display_df['test_start'].dt.strftime('%Y-%m-%d')
        display_df['test_end'] = display_df['test_end'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Performance by year
        st.subheader("Performance by Year")
        
        results_df["year"] = results_df["test_start"].dt.year
        yearly_perf = results_df.groupby("year")[["auc", "accuracy", "precision", "recall", "f1", "log_loss"]].mean()
        
        st.dataframe(yearly_perf, use_container_width=True)
        
        # Performance Metrics Over Time
        st.subheader("📈 Performance Metrics Over Time")
        
        # Create multi-line chart for all metrics
        fig_metrics = make_subplots(
            rows=2, cols=2,
            subplot_titles=('AUC Over Time', 'Accuracy Over Time', 
                          'Precision & Recall Over Time', 'F1 Score Over Time')
        )
        
        # AUC
        fig_metrics.add_trace(
            go.Scatter(x=results_df['test_start'], y=results_df['auc'],
                      mode='lines+markers', name='AUC', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Accuracy
        fig_metrics.add_trace(
            go.Scatter(x=results_df['test_start'], y=results_df['accuracy'],
                      mode='lines+markers', name='Accuracy', line=dict(color='green')),
            row=1, col=2
        )
        
        # Precision & Recall
        fig_metrics.add_trace(
            go.Scatter(x=results_df['test_start'], y=results_df['precision'],
                      mode='lines+markers', name='Precision', line=dict(color='orange')),
            row=2, col=1
        )
        fig_metrics.add_trace(
            go.Scatter(x=results_df['test_start'], y=results_df['recall'],
                      mode='lines+markers', name='Recall', line=dict(color='red')),
            row=2, col=1
        )
        
        # F1 Score
        fig_metrics.add_trace(
            go.Scatter(x=results_df['test_start'], y=results_df['f1'],
                      mode='lines+markers', name='F1', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig_metrics.update_xaxes(title_text="Date", row=2, col=1)
        fig_metrics.update_xaxes(title_text="Date", row=2, col=2)
        fig_metrics.update_yaxes(title_text="Score", row=1, col=1)
        fig_metrics.update_yaxes(title_text="Score", row=1, col=2)
        fig_metrics.update_yaxes(title_text="Score", row=2, col=1)
        fig_metrics.update_yaxes(title_text="Score", row=2, col=2)
        
        fig_metrics.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Actual vs Predicted Labels Over Time
        st.subheader("🎯 Actual vs Predicted Labels Over Time")
        
        # Aggregate by date
        daily_labels = predictions_df.groupby('Date').agg({
            'y_test': 'mean',
            'y_pred': 'mean'
        }).reset_index()
        
        daily_labels = daily_labels.sort_values('Date')
        
        fig_labels = go.Figure()
        
        fig_labels.add_trace(go.Scatter(
            x=daily_labels['Date'],
            y=daily_labels['y_test'],
            mode='lines',
            name='Actual Labels (avg)',
            line=dict(color='blue', width=2)
        ))
        
        fig_labels.add_trace(go.Scatter(
            x=daily_labels['Date'],
            y=daily_labels['y_pred'],
            mode='lines',
            name='Predicted Labels (avg)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_labels.update_layout(
            title='Average Actual vs Predicted Labels by Date',
            xaxis_title='Date',
            yaxis_title='Label (0=Down, 1=Up)',
            height=400,
            hovermode='x unified',
            xaxis=dict(
                range=['2022-02-01', '2023-01-31']
            )
        )
        st.plotly_chart(fig_labels, use_container_width=True)
        
        st.info("📝 Note: Values represent the average label across all stocks on each date. Values closer to 1 indicate more 'Up' predictions/actuals, while values closer to 0 indicate more 'Down' predictions/actuals.")
        
        # Stock-wise Performance
        st.subheader("📊 Stock-wise Performance Analysis")
        
        # Calculate metrics per stock
        stock_metrics = []
        for stock in predictions_df['Stock_symbol'].unique():
            stock_data = predictions_df[predictions_df['Stock_symbol'] == stock]
            
            if len(stock_data) > 0:
                y_true = stock_data['y_test'].values
                y_pred = stock_data['y_pred'].values
                
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                stock_metrics.append({
                    'Stock': stock,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1': f1
                })
        
        stock_perf_df = pd.DataFrame(stock_metrics)
        stock_perf_df = stock_perf_df.sort_values('Accuracy', ascending=False)
        
        # Display top and bottom performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🏆 Top 10 Performing Stocks (by Accuracy)**")
            st.dataframe(
                stock_perf_df.head(10).style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1': '{:.4f}'
                }),
                use_container_width=True,
                height=400
            )
        
        with col2:
            st.write("**⚠️ Bottom 10 Performing Stocks (by Accuracy)**")
            st.dataframe(
                stock_perf_df.tail(10).style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1': '{:.4f}'
                }),
                use_container_width=True,
                height=400
            )
        
        # Full stock performance table
        with st.expander("📋 View All Stocks Performance"):
            st.dataframe(
                stock_perf_df.style.format({
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1': '{:.4f}'
                }),
                use_container_width=True,
                height=600
            )
        
        # Download results
        csv_results = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_results,
            file_name=f"backtest_results_{selected_category}.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Select a stock category and click 'Run Backtest' to start the analysis")
