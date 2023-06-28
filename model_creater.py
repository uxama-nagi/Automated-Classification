import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
import joblib
import processing
warnings.filterwarnings("ignore")

# Dictionary of regular expressions related to each performance metric
regex_dict = {
"cpu usage": re.compile(r"\b(performance(?=\s*response|cpu|load|processor|utilization|usage)|cpu|load|processor|utilization|usage)\b", re.IGNORECASE),
"response time": re.compile(r"\b(performance(?=\s*response|optimize|latency|ping|laggy|lag|slow)|response(?=\s*rate)|response(?=\s*time)|execution(?=\s*time)|time(?=\s*execution)|time(?=\s*response)|optimize|latency|ping|laggy|lag|slow)\b", re.IGNORECASE),
"throughput": re.compile(r"\b(performance(?=\s*throughput|bandwidth|transaction)|throughput|bandwidth|transaction)\b", re.IGNORECASE),
"disk i/o": re.compile(r"\b(performance(?=\s*disk|read|write|storage)|io|disk(?=\s*input)|disk(?=\s*output)|read|write|storage)\b", re.IGNORECASE),
"memory usage": re.compile(r"\b(performance(?=\s*usage|consumption|ram|memory|hang)|usage|consumption|ram|memory|hang)\b", re.IGNORECASE),
}

vectorizer = TfidfVectorizer()

def per_msg_ext(df, col_name):
    print("\n...........Loading dataset...........")
    df = processing.read_csv_files(df)
    df = processing.preprocessing(df, col_name)
    df = processing.clean_stop_words(df,col_name)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)

    print("...........Extracting Performance Related Messages...........")
    # Performance issues keywords
    performance_issues = ['performance', 'latency', 'throughput', 'optimize', 'benchmark', 'slow', 'laggy', 'hang', 'response']

    # Binary Classification
    df['performance_issue'] = df[col_name+"_"+"cleaned"].apply(lambda x: 1 if any(issue in x for issue in performance_issues) else 0)
    df = df[df["performance_issue"] == 1]

    return df
    
def plot_distribution(dataset):
    # If labels are single values
    if isinstance(dataset[0], str):
        type_counts = dataset.value_counts()
        # Plot a pie chart of the most common types of performance issues
        plt.figure(figsize=(10, 5))
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        plt.title("Distribution of the dataset")
        plt.tight_layout()
        plt.savefig("distribution.png")
    # If labels are lists
    elif isinstance(dataset[0], list):
        all_labels = [label for sublist in dataset for label in sublist]
        type_counts = np.unique(all_labels, return_counts=True)
        # Plot a pie chart of the most common types of performance issues
        plt.figure(figsize=(10, 5))
        plt.pie(type_counts[1], labels=type_counts[0], autopct='%1.1f%%')
        plt.title("Distribution of the dataset")
        plt.tight_layout()
        plt.savefig("distribution.png")


def train_model(df, col_name):
    
    new_df = per_msg_ext(df, col_name)

    print("\n...........Labeling dataset...........")

    # Define a function to classify a message based on the matching regular expression
    def classify_message(message):
        for metric, regex in regex_dict.items():
            if regex.search(message):
                return metric
        return "Other"
    
    new_df["label"] = new_df[col_name+"_"+"cleaned"].apply(classify_message)

    plot_distribution(new_df['label'])

    print("\n...........Creating Model...........")
    y = new_df["label"].values
    X = new_df[col_name+"_"+"cleaned"].values

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Extract features from the text data
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    # Initialize different models
    pr = Perceptron(random_state=42)
    nb = MultinomialNB()

    best_model = None
    best_accuracy = 0.0

    models = [pr, nb]

    for model in models:
        # Fit the model and predict on the test set
        model.fit(X_train_features, y_train)
        y_pred = model.predict(X_test_features)
        
        # Calculate and print the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        print("\n")
        print(f'{model.__class__.__name__}: {accuracy:.3f}')
        print(classification_report(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f'\nBest model: {best_model.__class__.__name__} ({best_accuracy:.3f})')
    return joblib.dump(best_model, 'best_model.pkl'), joblib.dump(vectorizer ,"vectorizer.pkl")


if __name__ == "__main__":
    data = "data/dataset.csv"
    column = "message"
    train_model(data, column)