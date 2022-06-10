import pandas as pd
import os
import pickle

from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


def read_data(path):
    df = pd.read_parquet(path)
    return df
    

def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

       
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return


def get_paths(date):
    train_date = datetime.now().strftime('%Y-%m') if date is None else format(datetime.strptime(date, '%Y-%m-%d')-relativedelta(months=2), '%Y-%m')
    val_date = datetime.now().strftime('%Y-%m') if date is None else format(datetime.strptime(date, '%Y-%m-%d')-relativedelta(months=1), '%Y-%m')
    train_path = f'../data/fhv_tripdata_{train_date}.parquet' 
    val_path = f'../data/fhv_tripdata_{val_date}.parquet'
    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(date=None):
    train_path, val_path = get_paths(date)
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)

    # save dictvectorizer and datasets
    dump_pickle(dv, os.path.join('./output', f'dv-{date}.pkl'))
    dump_pickle(lr, os.path.join('./output', f'model-{date}.pkl'))



from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

# main(date="2021-08-15")

DeploymentSpec(
    flow=main(date="2021-08-15"),
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)


