import argparse
import pandas as pd

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--type', choices=['train', 'val'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if args.type == 'train':
        df = read_dataframe(args.year, args.month)
    else:
        next_year = args.year if args.month < 12 else args.year + 1
        next_month = args.month + 1 if args.month < 12 else 1
        df = read_dataframe(next_year, next_month)

    df.to_parquet(args.output)