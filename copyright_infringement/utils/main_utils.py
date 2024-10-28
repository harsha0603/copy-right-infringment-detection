import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import logging

# Initialize necessary resources for text processing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str, lower_case: bool = True, remove_special_chars: bool = True, 
                   remove_stopwords: bool = True, stemming: bool = False, lemmatization: bool = False) -> str:
    """
    Preprocesses a single string of text based on specified options.
    
    Parameters:
        text (str): The input text to preprocess.
        lower_case (bool): Whether to convert text to lowercase.
        remove_special_chars (bool): Whether to remove special characters.
        remove_stopwords (bool): Whether to remove stopwords.
        stemming (bool): Whether to apply stemming.
        lemmatization (bool): Whether to apply lemmatization.
    
    Returns:
        str: The preprocessed text.
    """
    try:
        # Lowercase the text if specified
        if lower_case:
            text = text.lower()

        # Remove special characters if specified
        if remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove stopwords if specified
        if remove_stopwords:
            text = ' '.join(word for word in text.split() if word not in stop_words)

        # Apply stemming if specified
        if stemming:
            text = ' '.join(stemmer.stem(word) for word in text.split())

        # Apply lemmatization if specified
        if lemmatization:
            text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())

        return text

    except Exception as e:
        logging.error(f"Error in preprocessing text: {e}")
        raise

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the given DataFrame.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame with possible duplicate rows.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    try:
        initial_length = len(data)
        data = data.drop_duplicates()
        final_length = len(data)
        logging.info(f"Removed {initial_length - final_length} duplicate entries.")
        return data

    except Exception as e:
        logging.error(f"Error in removing duplicates: {e}")
        raise

def get_user_input() -> str:
    """
    Prompts the user for input in a CLI environment. For Flask or web applications, 
    this can be adapted to handle input from a JSON request.
    
    Returns:
        str: User-provided input text.
    """
    try:
        # CLI version of user input; for Flask, use `request.json.get("text")` in the route
        user_input = input("Please enter text to check for similarity: ")
        return user_input

    except Exception as e:
        logging.error(f"Error in getting user input: {e}")
        raise

import pdfplumber
import os
from werkzeug.utils import secure_filename
import logging

def extract_text_from_pdf(pdf_file_path: str) -> str:
    text_content = ""
    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() or ""
        return text_content
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

def save_pdf(file, upload_folder="uploads"):
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(upload_folder, filename)
    file.save(pdf_path)
    return pdf_path
