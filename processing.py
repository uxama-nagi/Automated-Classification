import os
from nltk.corpus import stopwords
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

# A Funtion to remove contractions, non-english characters, multiple spaces and hyperlinks etc
def clean(msg):
    # Remove Java code
    msg = re.sub(r'```java[\s\S]*?```', ' ', str(msg))
    msg = re.sub(r'`{3}java[\s\S]*?`{3}', ' ', str(msg))
    # to avoid removing contractions in english
    msg = re.sub("'", "", str(msg)) 
    # Removing punctuations
    msg = re.sub(r'[()!?]', ' ', str(msg))
    msg = re.sub(r'\[.*?\]', ' ', str(msg))
    # Removing links
    msg = re.sub(r'^https?:\/\/.*[\r\n]*', '', str(msg), flags=re.MULTILINE)
    msg = re.sub(r'http\S+', '', str(msg))
    msg = re.sub(r"[^a-zA-Z ]", " ", str(msg))
    msg = re.sub(r' +', ' ', str(msg))
    msg = re.sub(r"\n", " ", str(msg))
    msg = re.sub(r"\r", " ", str(msg))
    return msg

def preprocessing(data, col_name):
    assert type(data)==pd.DataFrame
    assert type(col_name)==str

    data[col_name] = data[col_name].str.lower()
    print("---------- Preprocessing")
    data[col_name+"_"+"cleaned"] = [clean(msg) for msg in data[col_name]]
    print("---------- Preprocessing Completed")
    return data

# A funtion to remove stopwords
def clean_stop_words(data, col_name):
    assert type(data)==pd.DataFrame
    assert type(col_name)==str
    
    print("---------- Cleaning stopwords")
    stop_words = set(stopwords.words('english'))
    def remove_stopwords(text):
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in stop_words])
    data[col_name+"_"+"cleaned"] = data[col_name+"_"+"cleaned"].apply(remove_stopwords)
    print("---------- Cleaning Completed")
    return data

# An automate funtion which can take a file or path and load the data into pandas
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
    
    return df