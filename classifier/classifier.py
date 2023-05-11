import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import prepro

model_path = None
vectorizer_path = None
csv_path = None
label_encoder_path = None

def classify_csv():
    if model_path is None or vectorizer_path is None or csv_path is None:
        print("Please select all three files")
        return
    
    df = prepro.read_csv_files(csv_path)
    print("\nPlease Wait ---- Classifing Messages ")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    messages = df["message"]
    messages_features = vectorizer.transform(messages)
    predicted_classes = model.predict(messages_features)
    df["predicted_class"] = predicted_classes
    print("\nClassification Completed")
    df.to_csv('single classified.csv', index=False, header=True)
    messagebox.showinfo("Classification Completed!", "Classification completed successfully.")

def multi_classify_csv():
    if model_path is None or vectorizer_path is None or csv_path is None or label_encoder_path is None:
        print("Please select all four files for multi classifcation")
        return
    
    df = prepro.read_csv_files(csv_path)
    print("\nPlease Wait ---- Classifying Messages ")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    messages = df["message"]
    messages_features = vectorizer.transform(messages)
    predicted_classes = model.predict(messages_features)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    df["predicted_label"] = predicted_labels
    print("\nClassification Completed")
    df.to_csv('multi classified.csv', index=False, header=True)
    messagebox.showinfo("Classification Completed!", "Classification completed successfully.")

def select_model():
    global model_path
    model_path = filedialog.askopenfilename(title="Select Model", filetypes=[("Pickle Files", "*.pkl")])
    print("Model file selected:", model_path)

def select_vectorizer():
    global vectorizer_path
    vectorizer_path = filedialog.askopenfilename(title="Select Vectorizer", filetypes=[("Pickle Files", "*.pkl")])
    print("Vectorizer file selected:", vectorizer_path)

def select_csv():
    global csv_path
    csv_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    print("CSV file selected:", csv_path)

def select_label_encoder():
    global label_encoder_path
    label_encoder_path = filedialog.askopenfilename(title="Select Label Encoder", filetypes=[("Pickle Files", "*.pkl")])
    print("Label Encoder file selected:", label_encoder_path)

    
root = tk.Tk()
root.title("Classifier")
root.geometry("450x250")

model_label = tk.Label(root, text="Select model file:")
model_label.place(x=10, y=10)

model_button = tk.Button(root, text="Select model", command=select_model)
model_button.place(x=300, y=10)

vectorizer_label = tk.Label(root, text="Select vectorizer file:")
vectorizer_label.place(x=10, y=40)

vectorizer_button = tk.Button(root, text="Select vectorizer", command=select_vectorizer)
vectorizer_button.place(x=300, y=40)

csv_label = tk.Label(root, text="Select CSV file to classify:")
csv_label.place(x=10, y=70)

csv_button = tk.Button(root, text="Select CSV file", command=select_csv)
csv_button.place(x=300, y=70)

or_label = tk.Label(root, text="If you want multi classification select label encoder file\notherwise click classify button")
or_label.place(x=40, y=120)

label_encoder_label = tk.Label(root, text="Select label encoder file:")
label_encoder_label.place(x=10, y=170)

label_encoder_button = tk.Button(root, text="Select label encoder", command=select_label_encoder)
label_encoder_button.place(x=300, y=170)

classify_button = tk.Button(root, text="Classify", command=classify_csv, bg="#009dff", fg="white")
classify_button.place(x=150, y=220)

multi_classify_button = tk.Button(root, text="Multi Classify", command=multi_classify_csv, bg="#009dff", fg="white")
multi_classify_button.place(x=210, y=220)

root.mainloop()
