import unittest
import tempfile
import pickle
import os

import os
import sys
import boto3
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from aws.utils import (download_s3_train_data, uploud_s3_model, remove_data_dir, create_paths, save_model)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory after testing
        self.test_dir.cleanup()

    def test_download_s3_train_data(self):
        # TODO: Replace with your own test parameters
        PATH = self.test_dir.name
        BUCKET_NAME = 'mlflow-experiments-bucket'
        KEY = 'data/'
        FILENAME = 'train.csv'

        # Call the function being tested
        download_s3_train_data(PATH, BUCKET_NAME, KEY, FILENAME)

        # Check that the file was downloaded and moved to the correct location
        self.assertTrue(os.path.exists(os.path.join(PATH, KEY, FILENAME)))

    def test_uploud_s3_model(self):
        s3 = boto3.resource('s3')
        # TODO: Replace with your own test parameters
        PATH = self.test_dir.name
        BUCKET_NAME = 'mlflow-experiments-bucket'
        KEY = 'data/'

        # Create the 'data' directory if it doesn't exist
        data_dir = os.path.join(PATH, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Create a test file to upload
        test_file = os.path.join(PATH, 'data', 'dt_model_test.pkl')
        with open(test_file, 'wb') as f:
            f.write(b'test data')
        
        # Check that the file exists
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Could not find file {test_file}")

        # Call the function being tested
        uploud_s3_model(PATH, BUCKET_NAME, KEY)

        # Check that the file was uploaded to the correct location
        s3_object = s3.Object(BUCKET_NAME, 'data/dt_model_test.pkl')
        self.assertTrue(s3_object.key == 'data/dt_model_test.pkl')
        
    def test_remove_data_dir(self):
        # TODO: Replace with your own test parameters
        PATH = self.test_dir.name

        # Create a test directory to remove
        test_dir = os.path.join(PATH, 'data')
        os.makedirs(test_dir)

        # Call the function being tested
        remove_data_dir(PATH)

        # Check that the directory was removed
        self.assertFalse(os.path.exists(test_dir))

    def test_create_paths(self):
        # TODO: Replace with your own test parameters
        ABS_PATH = self.test_dir.name

        # Call the function being tested
        result = create_paths(ABS_PATH)

        # Check that the function returned the expected values
        self.assertEqual(len(result), 4)

        # Construct the expected file paths
        data_path = f"{ABS_PATH}/data"
        train_path = f"{data_path}/train.csv"
        preprocessing_path = f"{data_path}/preprocessing.csv"
        feature_eng_path = f"{data_path}/feature_engineering.csv"

        expected_paths = [train_path, preprocessing_path, feature_eng_path, data_path]

        for path in result:
            print(path)
            self.assertTrue(path in expected_paths)        

    def test_save_model(self):
        # TODO: Replace with your own test parameters
        model = {'test': 'data'}
        test_file = os.path.join(self.test_dir.name, 'dt_model_test.pkl')

        # Call the function being tested
        save_model(model, test_file)

        # Check that the model was saved to the correct file
        with open(test_file, 'rb') as f:
            loaded_model = pickle.load(f)
        self.assertEqual(model, loaded_model)
