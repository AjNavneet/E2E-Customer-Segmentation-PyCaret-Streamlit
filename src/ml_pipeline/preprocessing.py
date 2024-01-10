# Importing required libraries
from pycaret.clustering import *
from ml_pipeline.utils import read_config

# Reading config file
config = read_config("path/config.yaml")

# Function for splitting the data into training and testing sets
def train_test_split(dataset):
    # Sample 95% of the data for training, and the remaining 5% for testing
    data = dataset.sample(frac=0.95, random_state=42).reset_index(drop=True)
    data_unseen = dataset.drop(data.index).reset_index(drop=True)
    
    # Display the shape of the split datasets
    print('\nAfter splitting the data the shape of  - ')
    print('Data for Modeling: ' + str(data.shape))
    print('Unseen Data For Predictions: ' + str(data_unseen.shape),"\n")
    
    return data, data_unseen

# Function for setting up the environment in pycaret
def setup_env(dataset_name, **kwargs):
    # Set up the PyCaret environment for the given dataset
    cust_exp = setup(data=dataset_name)
    return cust_exp
