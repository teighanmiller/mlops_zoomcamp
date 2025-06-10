import pickle
from pathlib import Path
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://localhost:3000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def create_X(df, dv=None):
    categorical = ['PU_DO'] 
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run():
        run = mlflow.active_run()

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params ={
            'learning_rate': 0.0575188576815762,
            'max_depth': 33,
            'min_child_weight': 3.0654071229380224,
            'objective': 'reg:linear',
            'reg_alpha': 0.006897613262011037,
            'reg_lambda': 0.06078079888314963,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = (root_mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        with open('models/preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow") 

        return run.info.run_id

def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_month = month[0] + str(int(month[1]) + 1) if int(month) < 12 else 1
    next_year = year if int(month) < 12 else year + 1

    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'

    y_train = df_train[target].values
    y_val = df_val[target].values

    return train_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, dv=dv)

if __name__ == "__main__":
    # Use argparse to get year and month from command line
    import argparse

    parser = argparse.ArgumentParser(description="Train a model to predict taxi trip duration.")
    parser.add_argument('--year', type=str, required=True, help="Year of the data to train on.")
    parser.add_argument('--month', type=str, required=True, help="Month of the data to train on.")
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open('run_id.txt', "w") as f:
        f.write(run_id)