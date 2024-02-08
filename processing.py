import os
from nltk.corpus import stopwords
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

# A Funtion to remove contractions, non-english characters, multiple spaces and hyperlinks etc
def clean(msg:str):
    # Remove Java code
    msg = re.sub(r'```java[\s\S]*?```', ' ', str(msg))
    msg = re.sub(r'``java[\s\S]*?``', ' ', str(msg))
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

def preprocessing(data:pd.DataFrame, col_name:str):
    data[col_name+"_"+"cleaned"] = data[col_name].str.lower()
    print("---------- Preprocessing")
    data[col_name+"_"+"cleaned"] = data[col_name+"_"+"cleaned"].apply(clean)
    print("---------- Preprocessing Completed")
    return data

# A funtion to remove stopwords
def remove_stopwords(text:str):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return ' '.join([word for word in words if word.lower() not in stop_words])

def clean_stop_words(data:pd.DataFrame, col_name:str):
    print("---------- Cleaning stopwords")
    data[col_name+"_"+"cleaned"] = data[col_name+"_"+"cleaned"].apply(remove_stopwords)
    print("---------- Cleaning Completed")
    return data

# An automate funtion which can take a file or path and load the data into pandas
def read_csv_files(filepath:str):
    if os.path.isdir(filepath):
        # If filepath is a directory, read all CSV files in directory
        csv_files = [os.path.join(filepath, f) for f in os.listdir(filepath) if f.endswith('.csv')]
        print("---------- Loading all csv files from provided directory")
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
        # df.to_csv('dataset/onefile data.csv', index=False)
    elif os.path.isfile(filepath) and filepath.endswith('.csv'):
        # If filepath is a single CSV file, read it into a DataFrame
        print("---------- Loading csv file from provided directory")
        df = pd.read_csv(filepath)
    else:
        raise ValueError('Filepath must be a directory containing CSV files or a single CSV file.')
    return df