import numpy as np
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
import joblib
import prepro
warnings.filterwarnings("ignore")

# Performance issues keywords
performance_issues = ['performance', 'latency', 'throughput', 'optimize', 'benchmark', 'slow', 'laggy', 'hang', 'response']
# Vectorizer for extract features
vectorizer = TfidfVectorizer()

def per_msg_ext(df):

    print("\n...........Loading Dataset...........")
    df = prepro.read_csv_files(df)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)

    print("\n...........Extracting Performance Related Messages...........")
    # Binary Classification
    df['performance_issue'] = df['message'].apply(lambda x: 1 if any(issue in x for issue in performance_issues) else 0)
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


def single_label(df):
    
    new_df = per_msg_ext(df)
    print("\n...........Labeling dataset...........")
    new_df['label'] = new_df['message'].apply(lambda x: next((issue for issue in performance_issues if f" {issue} " in f" {x}"), 'None'))

    plot_distribution(new_df['label'])

    print("\n...........Creating Model...........")
    y = new_df["label"].values
    X = new_df['message'].values

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

# Funtion for Multi Labeling e.g. [slow, response, optimize]
def multi_label(df):

    new_df = per_msg_ext(df)

    print("\n...........Labeling dataset...........")
    
    new_df['label'] = new_df['message'].apply(lambda x: [issue for issue in performance_issues if f" {issue} " in f" {x}"])

    plot_distribution(new_df['label'])

    print("\n...........Creating Model...........")
    X = vectorizer.fit_transform(new_df['message'])

    label_encoder = MultiLabelBinarizer()
    y = label_encoder.fit_transform(new_df['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = OneVsRestClassifier(Perceptron())
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n")
    print(f'"Accuracy": {accuracy:.3f}')
    print(classification_report(y_test, y_pred))

    if not os.path.exists("multi"):
        os.makedirs("multi")
    return joblib.dump(classifier, 'mutli/multi_classifier.pkl'), joblib.dump(vectorizer ,"multi/vectorizer.pkl"), joblib.dump(label_encoder, "multi/label_encoder.pkl")


if __name__ == "__main__":
    data = "dataset.csv"
    function_to_run = sys.argv[1] if len(sys.argv) > 1 else None
    
    if function_to_run == "single":
        single_label(data)
    elif function_to_run == "multi":
        multi_label(data)
    else:
        print("Please specify either 'single' or 'multi' as an argument to run the corresponding function.")
