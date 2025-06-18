import argparse
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import mlflow

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with preprocessed data')
    parser.add_argument('--train_parquet', type=str, required=True, help="Train parquet for targets")
    parser.add_argument('--val_parquet', type=str, required=True, help='Val parquet for targets')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save models and artifacts')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Looking for: ", input_dir / "X_train.pkl")

    with open(input_dir / 'X_train.pkl', 'rb') as f_in:
        X_train = pickle.load(f_in)
    with open(input_dir / 'X_val.pkl', 'rb') as f_in:
        X_val = pickle.load(f_in)
    with open(input_dir / 'dv.pkl', 'rb') as f_in:
        dv = pickle.load(f_in)

    df_train = pd.read_parquet(args.train_parquet)
    df_val = pd.read_parquet(args.val_parquet)
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")

    # Save run_id to file
    with open(output_dir / "run_id.txt", "w") as f:
        f.write(run_id)
