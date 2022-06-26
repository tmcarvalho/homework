#!/usr/bin/env python
# coding: utf-8

import os
import sys

import uuid

import pickle
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


def generate_uuids(n):
    ride_ids = []
    for i in range(n):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids



def read_data(filename): 
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    df['ride_id'] = generate_uuids(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    dicts = df[categorical].to_dict(orient='records')

    return dicts


def get_paths(run_date):
    year = run_date.year
    month = run_date.month 
    
    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{year:04d}-{month:02d}.parquet'
    
    return input_file, output_file


def apply_model(input_file, output_file):
    df = read_data(input_file)
    dicts = dv.transform(prepare_dictionaries(df))
    y_pred = lr.predict(dicts)

    print(y_pred.mean())

    save_results(df, y_pred, output_file)
    return output_file


def ride_duration_prediction(run_date: datetime = None):
    
    input_file, output_file = get_paths(run_date)
    
    apply_model(
        input_file=input_file,
        output_file=output_file
    )


def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['PUlocationID'] = df['PUlocationID']
    df_result['DOlocationID'] = df['DOlocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']
    
    df_result.to_parquet(output_file, index=False)


def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2]) 

    ride_duration_prediction(
        run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == '__main__':
    run()
