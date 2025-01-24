import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists to store log files
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't already exist

# Set up logger for logging the feature engineering process
logger = logging.getLogger('feature_engineering')  # Create a logger named 'feature_engineering'
logger.setLevel('DEBUG')  # Set the logging level to DEBUG to capture detailed logs

# Define console handler to display logs in the console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Capture logs of level DEBUG and above

# Define file handler to store logs in a file
log_file_path = os.path.join(log_dir, 'feature_engineering.log')  # Define log file path
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')  # Capture logs of level DEBUG and above

# Set up log message format (timestamp, logger name, log level, message)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)  # Apply formatter to the console handler
file_handler.setFormatter(formatter)  # Apply formatter to the file handler

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """
    Load parameters from a YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)  # Parse YAML content
        logger.debug('Parameters retrieved from %s', params_path)  # Log successful parameter retrieval
        return params  # Return the parameters as a dictionary
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)  # Log if the file is not found
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)  # Log YAML parsing errors
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)  # Log any unexpected errors
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file and fill missing values.
    """
    try:
        df = pd.read_csv(file_path)  # Load data into a DataFrame
        df.fillna('', inplace=True)  # Replace missing values with an empty string
        logger.debug('Data loaded and NaNs filled from %s', file_path)  # Log successful data loading
        return df  # Return the loaded DataFrame
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)  # Log CSV parsing errors
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)  # Log other errors
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TfIdf (Term Frequency-Inverse Document Frequency) to transform the text data.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)  # Initialize TF-IDF vectorizer with max_features parameter

        # Extract the 'text' and 'target' columns for training and test data
        X_train = train_data['text'].values  # Training text data
        y_train = train_data['target'].values  # Training target labels
        X_test = test_data['text'].values  # Test text data
        y_test = test_data['target'].values  # Test target labels

        # Fit the vectorizer on the training data and transform both training and test data
        X_train_bow = vectorizer.fit_transform(X_train)  # Apply TF-IDF transformation to training data
        X_test_bow = vectorizer.transform(X_test)  # Apply the same transformation to test data

        # Convert the transformed data into DataFrames and include the target labels
        train_df = pd.DataFrame(X_train_bow.toarray())  # Convert train TF-IDF matrix to DataFrame
        train_df['label'] = y_train  # Add target labels to the training DataFrame

        test_df = pd.DataFrame(X_test_bow.toarray())  # Convert test TF-IDF matrix to DataFrame
        test_df['label'] = y_test  # Add target labels to the test DataFrame

        logger.debug('TfIdf applied and data transformed')  # Log successful TF-IDF transformation
        return train_df, test_df  # Return the transformed training and test DataFrames
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)  # Log errors during TF-IDF transformation
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save the transformed DataFrame to a CSV file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create the directory if it doesn't exist
        df.to_csv(file_path, index=False)  # Save the DataFrame to the specified file path without index
        logger.debug('Data saved to %s', file_path)  # Log successful data saving
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)  # Log errors during saving
        raise


def main():
    """
    Main function to load data, apply TF-IDF, and save the processed data.
    """
    try:
        # Step 1: Load parameters from the YAML configuration file
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']  # Retrieve the 'max_features' parameter
        
        # Step 2: Load the processed training and test data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        # Step 3: Apply TF-IDF transformation to the data
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        # Step 4: Save the transformed training and test data
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))

    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)  # Log any errors that occur
        print(f"Error: {e}")  # Print the error message


# Ensure that the main function is executed when the script is run directly
if __name__ == '__main__':
    main()
