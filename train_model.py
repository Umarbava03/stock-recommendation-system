from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE= Path("data/training_data.csv")
MODEL_DIR=Path("models")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_FILE=MODEL_DIR/"best_regressor.joblib"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "feature_importances.csv"

FEATURE_COLUMNS = [
    "price_change_pct",
    "volatility",
    "avg_volume",
    "ma_5",
    "ma_10",
    "trend_signal",
    "ma_20",
    "momentum_5",
    "volatility_5",
    "price_vs_ma_10",
    "rsi_14",
    "macd",
    "bollinger_width",
    "atr_14",
    "volume_spike_ratio",
    "market_return_5",
    "relative_strength_5",
    "avg_sentiment",
    "market_cap",
    "pe_ratio",
    "eps",
]

TARGET_COLUMN="target_return"

def load_data()->pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"The file not found at {DATA_FILE}")
    
    df=pd.read_csv(DATA_FILE)
    df=pd.DataFrame(df)
    return df

def train_model()-> None:
    df=load_data()
    train_split=int(len(df)*0.8)
    df["as_of_date"]=pd.to_datetime(df["as_of_date"])
    df=df.sort_values("as_of_date").reset_index(drop=True)
    x_train=df[FEATURE_COLUMNS].iloc[:train_split]
    x_test=df[FEATURE_COLUMNS].iloc[train_split:]
    y_train=df[TARGET_COLUMN].iloc[:train_split]
    y_test=df[TARGET_COLUMN].iloc[train_split:]


    models={"RandomForestRegressor":RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1),
        "LinearRegression":LinearRegression(),
        "GradientBoostingRegressor":GradientBoostingRegressor(random_state=42)
    }
    
    best_rmse=float("inf")
    best_model=None
    best_model_name=None
    best_predictions = None
    results=[]

    for model_name,model in models.items():
        model.fit(x_train,y_train)
        y_pred= model.predict(x_test)

        rmse=mean_squared_error(y_test,y_pred)**0.5
        mae=mean_absolute_error(y_test,y_pred)
        r2=r2_score(y_test,y_pred)
        pred_direction = y_pred > 0
        actual_direction = y_test > 0
        directional_accuracy = (pred_direction == actual_direction).mean()

        results.append(
            {
            "model":model_name,
            "mae":mae,
            "rmse":rmse,
            "R2":r2,
            "directional_accuracy": directional_accuracy
        })

        if rmse<best_rmse:
            best_model=model
            best_rmse=rmse
            best_mae=mae
            best_model_name=model_name
            best_r2=r2
            best_directional_accuracy=directional_accuracy
            best_predictions = y_pred


    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\nModel Comparison")
    print(results_df)


    if hasattr(best_model, "feature_importances_"):
        feature_importance_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        feature_importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
        print("\nFeature Importances")
        print(feature_importance_df)
    elif hasattr(best_model, "coef_"):
        coefficient_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "coefficient": best_model.coef_,
            }
        ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)

        coefficient_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
        print("\nFeature Coefficients")
        print(coefficient_df)



    print("Model Evaluation")
    print(f"MAE:  {best_mae:.6f}")
    print(f"RMSE: {best_rmse:.6f}")
    print(f"R2:   {best_r2:.6f}")
    print(f"Directional Accuracy: {best_directional_accuracy:.4f}")

    if best_predictions is not None:
        buy_mask = best_predictions > 0.01
        not_buy_mask = best_predictions < -0.02
        hold_mask = (~buy_mask) & (~not_buy_mask)

        buy_signal_count = int(buy_mask.sum())
        not_buy_signal_count = int(not_buy_mask.sum())
        hold_signal_count = int(hold_mask.sum())

        buy_signal_accuracy = (
            (y_test[buy_mask] > 0).mean() if buy_signal_count > 0 else 0.0
        )
        not_buy_signal_accuracy = (
            (y_test[not_buy_mask] < 0).mean() if not_buy_signal_count > 0 else 0.0
        )

        print("\nActionable Signal Evaluation")
        print(f"BUY signals: {buy_signal_count}")
        print(f"NOT BUY signals: {not_buy_signal_count}")
        print(f"HOLD signals: {hold_signal_count}")
        print(f"BUY signal accuracy: {buy_signal_accuracy:.4f}")
        print(f"NOT BUY signal accuracy: {not_buy_signal_accuracy:.4f}")
        pred_series = pd.Series(best_predictions)
        print(pred_series.describe())
        print(pred_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))



    joblib.dump(best_model, MODEL_FILE)
    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {MODEL_FILE}")



if __name__ == "__main__":
    train_model()
