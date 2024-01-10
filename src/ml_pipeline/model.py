# Importing required libraries
from pycaret.clustering import *
import pickle

# Building kmeans clustering model
def build_model(model_name, no_of_clusters):
    # Create a KMeans clustering model with the specified number of clusters
    kmeans = create_model('kmeans', no_of_clusters)
    return kmeans

# Assigning the cluster labels to our dataset
def assign_cluster_model(model_var_name):
    # Assign cluster labels to the dataset using the provided model
    kmean_results = assign_model(model_var_name)
    return kmean_results

# Function to save the final model
def save_final_model(model, model_path):
    # Append a directory and file name to the model path
    model_path += '/Final_kmeans_model'
    # Save the model using the specified path
    save_model(model, model_path)

# Function to load a pre-saved model
def load_presaved_model(model_path):
    # Append a directory and file name to the model path
    model_path += '/Final_kmeans_model'
    # Load the pre-saved model using the specified path
    return load_model(model_path)
