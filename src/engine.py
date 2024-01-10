# Importing required libraries
import pandas as pd
from pycaret.clustering import *
from yaml import CLoader as Loader, load

from ml_pipeline.utils import get_dataset, read_config, get_data_shape, save_predictions
from ml_pipeline.preprocessing import train_test_split, setup_env
from ml_pipeline.model import build_model, assign_cluster_model, save_final_model, load_presaved_model

# Reading config file
config = read_config("path/config.yaml")

# Getting the dataset
print("Getting the dataset - \n")
jewel_data = get_dataset(config['dataset'])

# Printing the shape of the dataset
get_data_shape(jewel_data)

# Splitting the data into training and testing sets
data, data_unseen = train_test_split(jewel_data)

# Setting up the environment in pycaret
cust_exp = setup_env(jewel_data)

# Prompting user to complete the setup
input("\n#### Please press Enter to complete the setup ####")

# Building kmeans clustering model
kmeans = build_model(config['model'], config['no_of_clusters'])

# Printing the model information
print("\nPrinting model information - \n", kmeans)

# Assigning the cluster labels to our dataset
kmean_results = assign_cluster_model(kmeans)

# Printing the results
print("\nThe top 5 rows of the dataset after being assigned labels - ")
print(kmean_results.head())

# Making predictions on the testing set
unseen_predictions = predict_model(kmeans, data=data_unseen)

# Saving the predictions as a CSV file
save_predictions(unseen_predictions, config['model_path'])

# Displaying the results of the testing set
print("Testing set predictions - \n", unseen_predictions.head())

# Saving the kmeans model
save_final_model(kmeans, config['model_path'])

# Loading the saved model
saved_kmeans = load_presaved_model(config['model_path'])

# Making predictions on the testing set again using the saved model
new_prediction = predict_model(saved_kmeans, data=data_unseen)

# Displaying the predictions
print("New Predictions - \n", new_prediction.head())
