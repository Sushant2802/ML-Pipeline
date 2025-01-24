import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk

# Download necessary NLTK datasets (stopwords and punkt tokenizer)
nltk.download('stopwords')
nltk.download('punkt')



# Ensure the "logs" directory exists for storing log files
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't already exist

# Setting up logger for logging the steps of data preprocessing
logger = logging.getLogger('data_preprocessing')  # Create a logger named 'data_preprocessing'
logger.setLevel('DEBUG')  # Set the logging level to DEBUG to capture detailed logs

# Define console handler to display logs in the console
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')  # Capture logs of level DEBUG and above

# Define file handler to store logs in a file
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')  # Define log file path
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')  # Capture logs of level DEBUG and above

# Set up log message format (timestamp, logger name, log level, message)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)  # Apply formatter to the console handler
file_handler.setFormatter(formatter)  # Apply formatter to the file handler

# Add both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, 
    removing stopwords and punctuation, and stemming the words.
    """
    ps = PorterStemmer()  # Initialize Porter Stemmer for stemming words
    
    # Step 1: Convert the text to lowercase
    text = text.lower()
    
    # Step 2: Tokenize the text into individual words
    text = nltk.word_tokenize(text)
    
    # Step 3: Remove non-alphanumeric tokens (numbers and special characters)
    text = [word for word in text if word.isalnum()]
    
    # Step 4: Remove stopwords (commonly used words that don't carry much meaning)
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Step 5: Stem the words (reduce them to their root form)
    text = [ps.stem(word) for word in text]
    
    # Step 6: Join the words back into a single string after processing
    return " ".join(text)


def preprocess_df(df, text_column='text', target_column='target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates,
    and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        
        # Step 1: Encode the target column (convert categorical labels to numerical)
        encoder = LabelEncoder()  # Initialize the LabelEncoder
        df[target_column] = encoder.fit_transform(df[target_column])  # Fit and transform the target column
        logger.debug('Target column encoded')  # Log after encoding
        
        # Step 2: Remove duplicate rows based on all columns
        df = df.drop_duplicates(keep='first')  # Keep the first occurrence of each duplicate row
        logger.debug('Duplicates removed')  # Log after removing duplicates
        
        # Step 3: Apply text transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)  # Apply the transform_text function
        logger.debug('Text column transformed')  # Log after text transformation
        
        return df  # Return the preprocessed DataFrame
    
    except KeyError as e:
        logger.error('Column not found: %s', e)  # Log if the specified column is not found
        raise  # Raise the error to stop further execution
    except Exception as e:
        logger.error('Error during text normalization: %s', e)  # Log other errors during preprocessing
        raise  # Raise the error to stop further execution




def main(text_column='text', target_column='target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Step 1: Load the raw data (train and test datasets)
        train_data = pd.read_csv('./data/raw/train.csv')  # Load the training data
        test_data = pd.read_csv('./data/raw/test.csv')  # Load the testing data
        logger.debug('Data loaded properly')  # Log successful data loading
        
        # Step 2: Preprocess the training and testing data
        train_processed_data = preprocess_df(train_data, text_column, target_column)  # Preprocess the train data
        test_processed_data = preprocess_df(test_data, text_column, target_column)  # Preprocess the test data
        
        # Step 3: Store the processed data inside the "data/processed" directory
        data_path = os.path.join("./data", "interim")  # Path to save the processed data
        os.makedirs(data_path, exist_ok=True)  # Create the directory if it doesn't exist
        
        # Save the processed data as CSV files
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed data saved to %s', data_path)  # Log after saving processed data
    
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)  # Log if the file is not found
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)  # Log if the CSV file is empty
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)  # Log any other errors
        print(f"Error: {e}")  # Print the error message


# Ensure that the main function is executed when the script is run directly
if __name__ == '__main__':
    main()
