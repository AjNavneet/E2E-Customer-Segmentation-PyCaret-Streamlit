# Importing required libraries
import pandas as pd
from pycaret.datasets import get_data
from pycaret.clustering import *
from yaml import CLoader as Loader, load

# Function for reading config file
def read_config(path):
    # Read and load the configuration from the specified YAML file
    with open(path) as stream:
        config = load(stream, Loader=Loader)
    return config

# Function to get data using PyCaret
def get_dataset(dataset_name):
    # Use PyCaret's get_data function to fetch the specified dataset
    jewel_data = get_data(dataset_name)
    return jewel_data

# Function to print the shape of the data
def get_data_shape(data):
    # Print the shape of the provided data
    print("\nThe shape of the data is ", data.shape)

# Function to save predictions as a CSV file
def save_predictions(predictions, output_path, **kwargs):
    # Append a file name to the output path and save predictions as a CSV file
    output_path += "/test.csv"
    predictions.to_csv(output_path, index=False)
