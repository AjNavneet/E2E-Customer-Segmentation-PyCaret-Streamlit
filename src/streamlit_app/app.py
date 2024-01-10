# Importing required libraries
from pycaret.clustering import *
import streamlit as st
import pandas as pd
from PIL import Image

# Loading the kmeans model
model = load_model('/output/Final_kmeans_model')

# Defining a function to make predictions
def predict(model, input_df):
    predictions_df = predict_model(model, data=input_df)
    predictions = predictions_df['Cluster'][0]
    return predictions

# Defining the main function
def run():

    # Loading an image
    image = Image.open('/customer_segmentation.png')

    # Adding the image to the web app
    st.image(image, use_column_width=True)

    # Adding a selectbox to make a choice between two prediction modes
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Online", "Batch"))

    # Adding some information about the app's functioning to the sidebar
    st.sidebar.info('This app is created to segment customers based on their behavior')

    # Adding the title for the Streamlit app
    st.title("Customer Segmentation Prediction App")

    # Adding steps to be followed if the user selects the "Online" mode of prediction 
    if add_selectbox == 'Online':

        # Adding number input boxes to get user input values
        Age = st.number_input('Age', min_value=18, max_value=100, value=25)
        Income = st.number_input('Income', min_value=9000, max_value=200000, value=20000)
        SpendingScore = st.number_input('SpendingScore', min_value=0.0 , max_value=1.0, format="%.2f")
        Savings = st.number_input('Savings', min_value=0.0 , max_value=25000.0, format="%.2f")

        # Defining the output variable 
        output=""

        # Creating an input dictionary with all the input features
        input_dict = {'Age': Age, 'Income': Income, 'SpendingScore': SpendingScore, 'Savings': Savings}

        # Converting the input dictionary into a pandas dataframe
        input_df = pd.DataFrame([input_dict])

        # Adding a button to make predictions when clicked on by the user
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)

        # Displaying the output after a successful prediction
        st.success('The output is {}'.format(output))

    # Adding steps to be followed if the user selects the "Batch" mode of prediction
    if add_selectbox == 'Batch':

        # Adding a file uploader button for the user to upload the CSV file containing data points
        file_upload = st.file_uploader("Upload CSV file for predictions", type=["csv"])

        # Block of code to be run once a CSV file is uploaded by the user
        if file_upload is not None:

            # Reading the CSV file using pandas
            data = pd.read_csv(file_upload)

            # Making predictions
            predictions = predict_model(model, data=data)

            # Displaying the predictions
            st.write(predictions)

# Calling the main function
if __name__ == '__main__':
    run()
