import shutil
import re
import os

import boto3
import botocore

import pickle

def download_s3_train_data(PATH, BUCKET_NAME, KEY, FILENAME):
    s3_client = boto3.client('s3')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    models = []
    
    for s3_object in bucket.objects.all():
        for key in bucket.objects.all():
            x = re.search("^data/*", key.key)
            if x:
                models.append(key.key)
    
    FOLDER = models[models.index(''.join([KEY, FILENAME]))]
    
    try:
        # s3_client.download_file(BUCKET_NAME, FOLDER, FILENAME)
        s3_client.download_file(BUCKET_NAME, FOLDER, os.path.join(PATH, FILENAME))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
    
    DIR_NAME = os.path.join(PATH, 'data')
    print(DIR_NAME)
    
    if not os.path.isdir(DIR_NAME):
       os.mkdir(DIR_NAME, 0o777)
        
    shutil.move(os.path.join(PATH, FILENAME), os.path.join(DIR_NAME, FILENAME))


def uploud_s3_model(PATH, BUCKET_NAME, KEY):
    client = boto3.client('s3')
    entries = os.listdir(f'{PATH}/data')
    filenames = [value for value in entries if re.search('^dt_model_*', value)]
    if filenames:
        filename = filenames[-1]
        client.upload_file(f"{PATH}/data/{filename}", BUCKET_NAME, f'{KEY}{filename}')
    else:
        print("No matching file found.")

def remove_data_dir(PATH):
    shutil.rmtree(PATH)

def create_paths(ABS_PATH):
    path = f'{ABS_PATH}/data'

    if not os.path.exists(path):
        os.makedirs(path)

    train_path = path + '/train.csv'
    preprocessing_path = path + '/preprocessing.csv'
    feature_eng_path = path + '/feature_engineering.csv'

    return train_path, preprocessing_path, feature_eng_path, path

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
