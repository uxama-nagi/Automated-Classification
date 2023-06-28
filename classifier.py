import tkinter as tk
import os
from tkinter import filedialog, messagebox
import joblib
import matplotlib.pyplot as plt
import processing
import sys
import threading


model_path = None
vectorizer_path = None
csv_path = None
column_name = "" 

###########################################
# Funtion to print the logs in a app screen
###########################################
def redirect_output(text_widget):
    class StdoutRedirector:
        def __init__(self, widget):
            self.widget = widget

        def write(self, message):
            self.widget.insert(tk.END, message)
            self.widget.see(tk.END)  # Automatically scroll to the end

    sys.stdout = StdoutRedirector(text_widget)

###############################################
# Funtion to generate the classification report
###############################################
def rep_calculate(data):
    # Group by repository and label, and calculate counts
    grouped_df = data.groupby(["repository", "predicted_class"]).size().unstack(fill_value=0)
    # Add a "Total" column
    grouped_df["Total"] = grouped_df.sum(axis=1)
    # Display the resulting dataset
    grouped_df.to_csv(os.getcwd()+"/output/report.csv", header=True )

##########################################################
# Funtion to plot the top then repositories classification
##########################################################
def plot_dis(data):
    # Group the data by repository and count the occurrences of each label
    grouped_data = data.groupby('repository')['predicted_class'].value_counts().unstack().fillna(0)
    # Get the top ten repositories based on the sum of all labels
    top_ten_repositories = grouped_data.sum(axis=1).nlargest(10).index
    # Filter the grouped data for the top ten repositories
    top_ten_data = grouped_data.loc[top_ten_repositories]
    # Plot the bar chart
    plt.figure(figsize=(20, 20))
    top_ten_data.plot(kind='bar', stacked=True)
    # Set the labels and title
    plt.xlabel('Repository')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.title('Top Ten Repositories and Label Distribution')
    plt.tight_layout()
    plt.savefig(os.getcwd()+"/output/top10.png", dpi=300)

############################
# Function to load a dataset
############################
def loaddata():
    df = processing.read_csv_files(csv_path)
    return df

#############################
# Function to load a csv file
#############################
def select_csv():
    global csv_path
    csv_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    print("\nCSV file:", csv_path)
    df = loaddata()
    print("\nDataset's Columns are:\n")
    print(df.columns)
    print('#' * 80)

##################################
# Function to load a trained model
##################################
def select_model():
    global model_path
    model_path = filedialog.askopenfilename(title="Select Model", filetypes=[("Pickle Files", "*.pkl")])
    print("\nModel file: ", model_path)

#######################################
# Function to load a trained vectorizer
#######################################
def select_vectorizer():
    global vectorizer_path
    vectorizer_path = filedialog.askopenfilename(title="Select Vectorizer", filetypes=[("Pickle Files", "*.pkl")])
    print("\nVectorizer file: ", vectorizer_path)

# ###################################################
# # Function to label the purpose of specific message 
# ###################################################

# # Define regular expressions to search for keywords
# bug_regex = re.compile(r'\b(bug|issue|error)\b', flags=re.IGNORECASE)
# # perf_regex = re.compile(r'\b(performance)\b', flags=re.IGNORECASE)
# fix_regex = re.compile(r'\b(fix|solution|solved)\b', flags=re.IGNORECASE)

# # Define a function to classify the message based on keywords
# def classify_message(msg, msg_type):
#     if msg_type == 'commit':
#         if bug_regex.search(msg) or fix_regex.search(msg):
#             return 'performence improvement'
#         # elif perf_regex.search(msg):
#         #     return 'performance improvement'
#         else:
#             return 'other'
#     elif msg_type == 'open_issue' or msg_type == 'closed_issue':
#         if bug_regex.search(msg):
#             return 'bug report'
#         # elif perf_regex.search(msg):
#         #     return 'performance issue'
#         elif fix_regex.search(msg):
#             return 'improvement needed'
#         else:
#             return 'other'
#     elif msg_type == 'open_pull_request' or  msg_type == 'closed_pull_request':
#         if bug_regex.search(msg) or fix_regex.search(msg):
#             return 'Solution Proposed'
#         # elif perf_regex.search(msg):
#         #     return 'performance issue'
#         # elif fix_regex.search(msg):
#         #     return 'improvement needed'
#         else:
#             return 'other'

########################################################################
# Function to check model, vectorizer and select column contains message 
########################################################################
def check_files(colname):
    if model_path is None or vectorizer_path is None:
        print("Please select all both files")
        return
    def select_column():
        # Create a new folder for outputs
        output_folder = "output"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        #################################
        df = loaddata()
        print("\n---------- Preparing Dataset")
        if colname in df.columns:
            df = processing.preprocessing(df, colname)
            df = processing.clean_stop_words(df, colname)
            print("---------- Classifing Messages")
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            messages = df[colname+"_"+"cleaned"]
            messages_features = vectorizer.transform(messages)
            predicted_classes = model.predict(messages_features)
            df["predicted_class"] = predicted_classes
            if 'repository' in df.columns:
                rep_calculate(df)
                plot_dis(df)
            print("\nClassification Completed Successfully")
            df = df.drop(colname+"_"+"cleaned", axis=1)
            df.to_csv(os.getcwd()+"/output/classified.csv", index=False, header=True)
            messagebox.showinfo("Classification Completed!", "Classification completed successfully.\n**Please find the outputs in C:\\Users\\(user)\\output**")
        else:
            print(f"\n{colname} not exist in the dataset")
            messagebox.showerror("Error Occured", f"{colname} not exist in the dataset")
            
    threading.Thread(target=select_column).start()

###################################
# Funtion to perform Classification
###################################
def classification():
    global column_name
    column_name = message_col_entry.get()
    check_files(column_name)
    if not column_name:
        messagebox.showerror("Error", "Please enter the Column Name")
        return

##########################
# UI using Tkinter Library
##########################  
root = tk.Tk()
root.title("Performance Issue Classifier")
root.geometry("640x680")
root.resizable(False, False)

logo = tk.PhotoImage(file = "C:\\Users\\Nagi\\Desktop\\thesis_scripts\\image\\logo.gif")
root.iconphoto(False, logo)

bg_img = tk.PhotoImage(file="C:\\Users\\Nagi\\Desktop\\thesis_scripts\\image\\image.png")
img_label = tk.Label(root, image = bg_img)
img_label.place(x=0, y=-5, width=650)

csv_label = tk.Label(root, text="Select CSV file to classify:")
csv_label.place(x=70, y=20)

csv_button = tk.Button(root, text="Select CSV file", command=select_csv)
csv_button.place(x=420, y=20)

message_col_label = tk.Label(root, text="Type column name which contains messages:")
message_col_label.place(x=70, y=60)

message_col_entry = tk.Entry(root)
message_col_entry.place(x=420, y=60)

model_label = tk.Label(root, text="Select model file:")
model_label.place(x=70, y=100)

model_button = tk.Button(root, text="Select model", command=select_model)
model_button.place(x=420, y=100)

vectorizer_label = tk.Label(root, text="Select vectorizer file:")
vectorizer_label.place(x=70, y=140)

vectorizer_button = tk.Button(root, text="Select vectorizer", command=select_vectorizer)
vectorizer_button.place(x=420, y=140)

classify_button = tk.Button(root, text="Classify", width=20 ,command=classification, bg="#009dff", fg="white")
classify_button.place(x=240, y=200)

# Text widget to display the print statements
output_text = tk.Text(root, bg="#01364a", fg="white")
output_text.place(x=0, y=240, height=800)


redirect_output(output_text)
print(" " * 37 + "Logs" + "\n" + "#" * 80)
root.mainloop()