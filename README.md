# Stock Market Recommendation System

This project is a hybrid stock analysis and recommendation system that combines:

- technical analysis
- fundamental analysis
- news sentiment analysis
- machine learning based return prediction

The system predicts a short-term future return for a stock and converts that prediction into a recommendation:

- `BUY`
- `HOLD / WATCH`
- `NOT BUY`

## Features

- Fetches recent stock price data
- Extracts technical indicators such as moving averages, volatility, momentum, and RSI
- Fetches company fundamentals such as market cap, P/E ratio, and EPS
- Analyzes news sentiment with ticker relevance weighting
- Trains a Random Forest regressor on historical stock data
- Compares ML prediction with a rule-based baseline
- Saves latest prediction output to JSON
- Saves feature importances for model interpretation

## Project Structure

```text
Final_year_project/
├── build_dataset.py
├── config.py
├── features.py
├── fundamentals.py
├── main.py
├── news_sentiment.py
├── predictor.py
├── stock_data.py
├── train_model.py
├── providers/
│   ├── alpha_vantage_provider.py
│   ├── fmp_provider.py
│   └── twelve_data_provider.py
├── cache/
├── data/
├── models/
└── README.md
```

## Data Sources

- **Twelve Data**: recent stock price data for live pipeline usage
- **Financial Modeling Prep (FMP)**: historical price data and fundamentals
- **Alpha Vantage**: news feed with ticker relevance metadata
- **NLTK VADER**: sentiment scoring for relevant news headlines

## Technical Features Used

- `price_change_pct`
- `volatility`
- `avg_volume`
- `ma_5`
- `ma_10`
- `ma_20`
- `trend_signal`
- `momentum_5`
- `volatility_5`
- `price_vs_ma_10`
- `rsi_14`

## Fundamental Features Used

- `market_cap`
- `pe_ratio`
- `eps`

## Sentiment Feature

- `avg_sentiment`

This is computed using:

- ticker relevance from Alpha Vantage
- VADER sentiment score on headlines
- weighted average sentiment across relevant articles

## Model

The current ML model is:

- `RandomForestRegressor`

Target variable:

- `target_return = (future_close - current_close) / current_close`

Recommendation logic:

- predicted return `> 3%` -> `BUY`
- predicted return `> 0% and <= 3%` -> `HOLD / WATCH`
- predicted return `<= 0%` -> `NOT BUY`

## Setup

Create and activate your virtual environment, then install required libraries.

Example dependencies include:

- `pandas`
- `requests`
- `scikit-learn`
- `joblib`
- `nltk`

You also need these environment variables:

```bash
export FMP_API_KEY="your_fmp_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
export TWELVE_DATA_API_KEY="your_twelve_data_key"
```

## How To Run

### 1. Build training dataset

```bash
python build_dataset.py
```

This creates:

- `data/training_data.csv`

### 2. Train the model

```bash
python train_model.py
```

This creates:

- `models/random_forest_regressor.joblib`
- `data/feature_importances.csv`

### 3. Run live prediction

```bash
python main.py --symbol AMZN
python main.py --symbol AAPL
python main.py --symbol MSFT
```

This creates:

- `data/latest_prediction.json`

### 4. Run the API

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api:app --reload
```

Example request:

```bash
curl "http://127.0.0.1:8000/predict?symbol=AMZN"
```

## Output

The program prints:

- extracted features
- ML-based recommendation
- predicted 5-day return
- confidence score
- rule-based recommendation for comparison

## Current Limitations

- historical sentiment is not time-aligned in the training dataset
- historical fundamentals are not time-varying in the dataset
- some APIs are rate-limited or plan-limited
- model performance can be improved with more features and broader data

## Future Improvements

- add more symbols and sectors
- include time-aligned historical sentiment
- include historical fundamental snapshots
- compare multiple regression models
- add LSTM or sequence-based price forecasting as a secondary model
- build a frontend dashboard for predictions and visualizations
