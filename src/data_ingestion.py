import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml


# Ensure the "logs" directory exists for storing log files
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't already exist


# Set up logging configuration for debugging
logger = logging.getLogger('data_ingestion')  # Create a logger named 'data_ingestion'
logger.setLevel('DEBUG')  # Set the log level to DEBUG to capture detailed logs


# Define console handler to display logs in the console
console_handler = logging.StreamHandler()  
console_handler.setLevel('DEBUG')  # Log messages of level DEBUG and above to console

# Define file handler to save logs to a file
log_file_path = os.path.join(log_dir, 'data_ingestion.log')  # Define the file path for logs
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')  # Log messages of level DEBUG and above to the file

# Set up a formatter to define the format of the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)  # Apply the formatter to the console handler
file_handler.setFormatter(formatter)  # Apply the formatter to the file handler

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)





# Function to load parameters from a YAML configuration file
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:  # Open the YAML file in read mode
            params = yaml.safe_load(file)  # Load the contents of the file into a dictionary
        logger.debug('Parameters retrieved from %s', params_path)  # Log successful retrieval
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)  # Log error if file is not found
        raise  # Raise the error to stop further execution
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)  # Log YAML parsing error
        raise  # Raise the error to stop further execution
    except Exception as e:
        logger.error('Unexpected error: %s', e)  # Log any other unexpected error
        raise  # Raise the error to stop further execution


# Function to load data from a CSV file (using a URL or file path)
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)  # Load the CSV file into a pandas DataFrame
        logger.debug('Data loaded from %s', data_url)  # Log successful loading of data
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)  # Log error if CSV parsing fails
        raise  # Raise the error to stop further execution
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)  # Log other errors
        raise  # Raise the error to stop further execution


# Function to preprocess the loaded data (e.g., cleaning and renaming columns)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # Drop unnecessary columns
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        
        # Rename columns for better clarity
        df.rename(columns = {'v1': 'target', 'v2': 'text'}, inplace = True)
        logger.debug('Data preprocessing completed')  # Log successful preprocessing
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)  # Log error if expected columns are missing
        raise  # Raise the error to stop further execution
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)  # Log other unexpected errors
        raise  # Raise the error to stop further execution


# Function to save the processed data (train and test splits) to a specified path
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        # Create a sub-directory 'raw' to store the data
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)  # Create the directory if it doesn't exist
        
        # Save the train and test datasets as CSV files
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)  # Log successful saving
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)  # Log any saving errors
        raise  # Raise the error to stop further execution




# Main function that coordinates the whole data ingestion process
def main():
    try:
        test_size = 0.2 
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        
        # Load the data from the specified URL
        df = load_data(data_url=data_path)
        
        # Preprocess the data (cleaning and renaming)
        final_df = preprocess_data(df)
        
        # Split the data into train and test sets using the specified test size
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save the train and test data to the specified path
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)  # Log the error if the process fails
        print(f"Error: {e}")  # Print the error to the user


# Ensure the main function is executed if the script is run directly
if __name__ == '__main__':
    main()
