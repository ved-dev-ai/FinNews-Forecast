# FinNews-Forecast 📈

A comprehensive machine learning pipeline for stock price movement prediction that combines financial news sentiment analysis with technical indicators. This project leverages FinBERT for sentiment analysis and XGBoost for predictive modeling across multiple market sectors.

## 🌟 Features

- **Multi-Source Data Collection**: Automated collection of stock price data and financial news
- **Sentiment Analysis**: FinBERT-powered sentiment scoring of financial news articles
- **Advanced Feature Engineering**: Technical indicators and sentiment-based features
- **Time-Series ML Pipeline**: Rolling window validation for robust model training
- **Multi-Sector Support**: Models for 12 different market sectors
- **Interactive Dashboard**: Streamlit-based web application for model backtesting and visualization
- **Scalable Architecture**: Modular design supporting both local and cloud deployments

## 📊 Supported Market Sectors

The pipeline supports prediction models for the following sectors:

- Communication Services
- Consumer Discretionary
- Consumer Staples
- Energy
- Financial Services
- Healthcare
- Industrials
- Materials
- Miscellaneous
- Real Estate
- Technology
- Utilities

## 🏗️ Project Structure

```
FinNews-Forecast/
├── Datasets/
│   ├── categorical_stocks_data/    # Parquet data partitioned by category
│   └── category_csvs/               # CSV exports by sector
├── models/                          # Trained models by sector
├── Notebooks/                       # Jupyter notebooks for exploration
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_finbert_data_preperation.ipynb
│   ├── 04_score_news_sentiments.ipynb
│   ├── 05_merge sentiment_data.ipynb
│   ├── 06_EDA.ipynb
│   └── 07_ML_pipeline_2.ipynb
├── Scripts/
│   ├── data_engineering/            # Data collection and processing
│   ├── machine_learning/            # ML pipeline scripts
│   ├── sentiment_analysis/          # FinBERT sentiment scoring
│   └── streamlit/                   # Web application
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FinNews-Forecast
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Running the Complete Pipeline

The project follows a sequential pipeline. Execute the scripts in order:

#### 1. Data Collection
```bash
python Scripts/data_engineering/01_data_collection.py
```
Collects historical stock price data and financial news articles.

#### 2. Data Cleaning
```bash
python Scripts/data_engineering/02_data_cleaning.py
```
Cleans and preprocesses raw data, handles missing values and outliers.

#### 3. FinBERT Data Preparation
```bash
python Scripts/data_engineering/03_finbert_data_preperation.py
```
Prepares news text data for sentiment analysis.

#### 4. Sentiment Scoring
```bash
python Scripts/sentiment_analysis/04_score_news_sentiments.py
```
Applies FinBERT model to score sentiment of financial news.

#### 5. Merge Sentiment Data
```bash
python Scripts/data_engineering/05_merge_sentiment_data.py
```
Combines sentiment scores with stock price data.

#### 6. Convert to CSV
```bash
python Scripts/data_engineering/06_data_parquet_to_csv.py
```
Exports processed data to CSV format by category.

#### 7. Train ML Models
```bash
python Scripts/machine_learning/07_ml_pipeline_v2.py
```
Trains XGBoost models with rolling window validation for each sector.

### Running the Interactive Dashboard

Launch the Streamlit web application for interactive backtesting:

**Local version:**
```bash
streamlit run Scripts/streamlit/streamlit_app_local.py
```

**Cloud version:**
```bash
streamlit run Scripts/streamlit/streamlit_app.py
```

The dashboard allows you to:
- Select different stock market sectors
- Run model backtests
- Visualize prediction results
- Analyze model performance metrics
- View confusion matrices and ROC curves

## 🔬 Exploratory Analysis

The `Notebooks/` directory contains Jupyter notebooks for each pipeline stage, enabling:
- Interactive data exploration
- Visualization of intermediate results
- Experimentation with different parameters
- Ad-hoc analysis and prototyping

To use the notebooks:
```bash
jupyter notebook Notebooks/
```

## 🤖 Model Details

### Features
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands, etc.
- **Sentiment Features**: FinBERT sentiment scores aggregated over different time windows
- **Price Action**: Returns, volatility, volume metrics

### Algorithm
- **Primary Model**: XGBoost Classifier
- **Validation**: Rolling window time-series cross-validation
- **Target**: Binary classification (price up/down)

### Evaluation Metrics
- ROC-AUC Score
- Accuracy
- Precision & Recall
- F1 Score
- Log Loss
- Confusion Matrix

## 📦 Dependencies

Key libraries used in this project:

- **Data Processing**: pandas, numpy, pyarrow
- **Machine Learning**: scikit-learn, xgboost
- **NLP/Sentiment**: transformers, torch (FinBERT)
- **Visualization**: plotly, streamlit
- **Cloud Storage**: boto3, minio
- **Web Scraping**: lxml
- **Utilities**: tqdm, huggingface_hub

See [requirements.txt](requirements.txt) for complete list.

## 🔧 Configuration

- Modify data sources in the data collection scripts
- Adjust model hyperparameters in the ML pipeline script
- Configure cloud storage credentials for remote deployments

## 📈 Performance

The models are evaluated using time-series cross-validation to ensure robustness and prevent data leakage. Performance varies by sector and market conditions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FinBERT model by ProsusAI
- XGBoost library developers
- Streamlit team for the amazing framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes only. Always conduct thorough due diligence before making any investment decisions.
