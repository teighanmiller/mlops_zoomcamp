import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', type=str, required=True)
    parser.add_argument('--val_input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train_input)
    df_val = pd.read_parquet(args.val_input)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    with open(output_dir / "X_train.pkl", "wb") as f_out:
        pickle.dump(X_train, f_out)

    with open(output_dir / "X_val.pkl", "wb") as f_out:
        pickle.dump(X_val, f_out)

    with open(output_dir / "dv.pkl", 'wb') as f_out:
        pickle.dump(dv, f_out)

