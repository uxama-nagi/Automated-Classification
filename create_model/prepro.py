import os
from nltk.corpus import stopwords
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

def pre_process(df):
    # A fuction to clean messages
    def clean(msg):
        # to avoid removing contractions in english
        msg = re.sub("'", "", str(msg)) 
        # Removing punctuations
        msg = re.sub(r'[()!?]', ' ', str(msg))
        msg = re.sub(r'\[.*?\]', ' ', str(msg))
        msg = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(msg), flags=re.MULTILINE)
        msg = re.sub(r'http\S+', '', str(msg))
        msg = re.sub(r"[^a-zA-Z ]", " ", str(msg))
        msg = re.sub(r' +', ' ', str(msg))
        msg = re.sub(r"\n", " ", str(msg))
        msg = re.sub(r"\r", " ", str(msg))
        return msg

    stop_words = set(stopwords.words('english'))
    def remove_stopwords(text):
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in stop_words])
    
    print("\n...........Pre-Processing Data...........")
    df['message'] = df['message'].str.lower()
    df["message"] = [clean(msg) for msg in df['message']]
    df['message'] = df['message'].apply(remove_stopwords)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['timestamp'].notnull()]
    df = df.reset_index(drop=True)
    print("\n...........Pre-Processing completed successfully...........")
    return df
    
def read_csv_files(filepath):
    if os.path.isdir(filepath):
        # If filepath is a directory, read all CSV files in directory
        csv_files = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.endswith('.csv')]
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    elif os.path.isfile(filepath) and filepath.endswith('.csv'):
        # If filepath is a single CSV file, read it into a DataFrame
        df = pd.read_csv(filepath)
    else:
        raise ValueError('Filepath must be a directory containing CSV files or a single CSV file.')
    
    pre_process(df)
    return df