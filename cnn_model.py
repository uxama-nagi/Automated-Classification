from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import processing
import re
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import pad_sequences
from sklearn.metrics import classification_report


performance_issues = ['performance', 'latency', 'throughput', 'optimize', 'benchmark', 'slow', 'laggy', 'hang', 'response']
regex_dict = {
"cpu usage": re.compile(r"\b(cpu|load|processor|utilization|usage)\b", re.IGNORECASE),
"response time": re.compile(r"\b(response|optimize|time|latency|ping|laggy|slow)\b", re.IGNORECASE),
"throughput": re.compile(r"\b(throughput|bandwidth|transaction)\b", re.IGNORECASE),
"disk i/o": re.compile(r"\b(disk|read|write|storage)\b", re.IGNORECASE),
"memory usage": re.compile(r"\b(usage|consumption|ram|memory|hang)\b", re.IGNORECASE),
}

def create_model(data, col_name):

    print("\n...........Loading dataset...........")
    df = processing.read_csv_files(data)
    df = processing.preprocessing(df, col_name)
    df = processing.clean_stop_words(df,col_name)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Label the data
    print("...........Extracting Performance Related Messages...........")
    df['performance_issue'] = df[col_name+"_"+"cleaned"].apply(lambda x: 1 if any(issue in x for issue in performance_issues) else 0)
    df = df[df["performance_issue"] == 1]

    # Define a function to classify a message based on the matching regular expression
    def classify_message(message):
        for metric, regex in regex_dict.items():
            if regex.search(message):
                return metric
        return "Other"

    df["label"] = df["message"].apply(classify_message)

    # Split the dataset into messages and labels
    messages = df[col_name+"_"+"cleaned"].values
    labels = df['label'].values

    # Perform label encoding on the labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Split the dataset into training and testing sets
    train_messages, test_messages, train_labels, test_labels = train_test_split(messages, labels, test_size=0.2, random_state=42)

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_messages)
    vocab_size = len(tokenizer.word_index) + 1

    # Convert text to sequences
    train_sequences = tokenizer.texts_to_sequences(train_messages)
    test_sequences = tokenizer.texts_to_sequences(test_messages)

    # Pad sequences to have a consistent length
    max_sequence_length = 100  # Set the desired sequence length
    train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
    test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

    # Build the CNN model
    embedding_dim = 100  # Set the desired embedding dimension
    num_filters = 128  # Set the number of filters in the convolutional layer
    filter_size = 5  # Set the size of the filters in the convolutional layer

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    model.add(Conv1D(num_filters, filter_size, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    batch_size = 32  # Set the batch size
    epochs = 10  # Set the number of training epochs

    history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))

    # Evaluate the model
    loss, accuracy = model.evaluate(test_data, test_labels)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Plot the loss and accuracy during training
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

    # Get the predictions
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    # Reverse label encoding for the original labels and predicted labels
    original_labels = label_encoder.inverse_transform(test_labels)
    predicted_labels1 = label_encoder.inverse_transform(predicted_labels)

    # Generate the classification report
    report = classification_report(original_labels, predicted_labels1)
    print(report)

    if not os.path.exists("cnn"):    
        os.makedirs("cnn")

    return joblib.dump(tokenizer, "cnn/vectorizer.pkl"), joblib.dump(model, "cnn/model.pkl"),joblib.dump(label_encoder, "cnn/label_encoder.pkl")


if __name__ == "__main__":
    data = "dataset.csv"
    column = "message"
    create_model(data)