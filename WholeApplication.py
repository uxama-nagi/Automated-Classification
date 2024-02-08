import csv
import os
import re
import threading
import time
import tkinter as tk
import warnings
from tkinter import filedialog, messagebox
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from github import Github, RateLimitExceededException
from pandastable import Table, Toplevel
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.utils import resample, shuffle
from helpers import *
import processing
warnings.filterwarnings("ignore")


statement = "#" * 43 + " Logs " + "#" * 45
#################################################################
#################### Main Screen ####################
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Performance Issue Classifier")
        self.geometry("640x400")
        self.resizable(False, False)

        logo = tk.PhotoImage(file=resource_path("logo.png"))
        self.iconphoto(False, logo)

        _canvas = tk.Canvas(self, width=640, height=400, bd=0, borderwidth=0)
        _canvas.pack()
        
        self.bg_img = tk.PhotoImage(file=resource_path("main2.png"))
        _canvas.create_image(0,0, image=self.bg_img, anchor="nw")

        self.button1 = tk.Button(self, text="Mine Data", command=self.open_screen1, bg="#0085E2", fg="white")
        self.button1.place(x=260, y=120, width=130, height=50)
        self.button2 = tk.Button(self, text="Create Model", command=self.open_screen2, bg="#0085E2", fg="white")
        self.button2.place(x=260, y=190, width=130, height=50)
        self.button3 = tk.Button(self, text="Perform Classification", command=self.open_screen3, bg="#0085E2", fg="white")
        self.button3.place(x=260, y=260, width=130, height=50)

        self.name_label = tk.Label(self, text="Muhammad Usama Bin Abad")
        self.name_label.place(x=475, y=380)
        self.dwnd = tk.PhotoImage(file=resource_path("qicon.png"))
        self.info = _canvas.create_image(592,20,  image=self.dwnd, anchor="nw")
        _canvas.tag_bind(self.info, "<Button-1>", self.open_info)

    def open_info(self, event):
        messagebox.showinfo("How to Use", "Press Mine Data button to mine data from Github.\n-\
                            \nPress Create Model button to train different models.\n-\
                            \nPress Perform Classification button if you already have a trained model and want to classify new data.")
    
    def open_screen1(self):
        Screen1()

    def open_screen2(self):
        Screen2()

    def open_screen3(self):
        Screen3()


#################################################################
########################## Mine Data ############################
class Screen1(tk.Toplevel):
    def __init__(self):
        super().__init__()
        
        self.query = ""
        self.access_token = ""
        self.keywords = ""
        self.output_dir = ""
        self.num_row_per_file = ""
        self.keywords_list = []
        self.keywords_path = None

        self.title("Performance Issue Classifier - Data Mining Window")
        self.geometry("640x520")
        self.resizable(False, False)
        # Logo and Background
        logo = tk.PhotoImage(file = resource_path("logo.png"))
        self.iconphoto(False, logo)

        _canvas = tk.Canvas(self, width=640, height=400, bd=0, borderwidth=0)
        _canvas.pack()

        self.bg_img = tk.PhotoImage(file= resource_path("image.png"))
        _canvas.create_image(0,0, image=self.bg_img, anchor="nw")
        # Label and Text Field for Query
        _canvas.create_text(70,22, text="Please write the Query", font=("Open Sans", 10), anchor="nw")
        self.query_entry = tk.Entry(self)
        self.query_entry.place(x=380, y=20, width=200)
        # Label and Text Field for Access Token
        _canvas.create_text(70,52, text="GitHub API Access Token", font=("Open Sans", 10), anchor="nw")
        self.api_entry = tk.Entry(self)
        self.api_entry.place(x=380, y=50, width=200)
        # Label and Text Field for output folder name
        _canvas.create_text(70,82, text="Please write the output folder name", font=("Open Sans", 10), anchor="nw")
        self.output_entry = tk.Entry(self)
        self.output_entry.place(x=380, y=80, width=200)
        # Label and Text Field for Number of Rows in each csv file
        _canvas.create_text(70,112, text="How many rows in each csv file to write 1-100000", font=("Open Sans", 10), anchor="nw")
        self.row_entry = tk.Entry(self)
        self.row_entry.place(x=380, y=110, width=200)
        # Label and Text Field for Keywords
        _canvas.create_text(70,142, text="Upload keywords file", font=("Open Sans", 10), anchor="nw")
        self.keywords_upload_button = tk.Button(self, text="Upload Keywords", command=self.upload_keywords)
        self.keywords_upload_button.place(x=380, y=140)
        # Button for Start mining
        self.mine_button = tk.Button(self, text="Start Mining", width=20 ,command=self.search_github, bg="#0085E2", fg="white")
        self.mine_button.place(x=240, y=200)

        self.info_button = tk.Button(self, text="Fill Sample Inputs", command=self.open_info) 
        self.info_button.place(x=20, y=200)

        # Text widget to display the print statements
        _canvas.create_text(0,243, text=statement, fill="#01364a", anchor="nw")
        self.output_text = tk.Text(self, bg="#01364a", fg="white")
        self.output_text.place(x=0, y=260, height=260)
        # Log area 
        redirect_output(self.output_text)
        self.grab_set()  # This will make the window modal (block access to main window)
    
    def upload_keywords(self):
        self.keywords_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        with open(self.keywords_path, 'r') as file:
            content = file.read()
        # Process the content to extract the keywords as a list
        self.keywords_list = content.strip('[]\n').split(', ')
        print("keywords file: ", self.keywords_path)
        print(f"keywords: {self.keywords_list}")        
    
    # Main Funtion to Mine data from GitHub
    def search_github(self):
        def parallel_mining():
            self.mine_button.config(state="disabled")
            try:
                # variables to store inputs
                self.query = self.query_entry.get()
                self.access_token = self.api_entry.get()
                self.keywords = self.keywords_list
                self.output_dir = self.output_entry.get()
                self.num_row_per_file = self.row_entry.get() 
                # Check all arguments
                if not self.query or not self.access_token or not self.keywords or not self.output_dir or not self.num_row_per_file:
                    messagebox.showerror("Error", "Please enter all arguments")
                    return
                if self.keywords_path is None or self.keywords == []:
                    messagebox.showerror("Error", "Please upload keyword file with some keywords")
                    return
                    
                # Access token to access the GitHub API
                g = Github(self.access_token, timeout=60)
                start = time.time()
                # Create the output directory if it doesn't exist
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

                # Open the first CSV file for writing
                file_count = 1
                max_rows_per_file = self.num_row_per_file
                csvfile = open(os.path.join(self.output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
                fieldnames = ['created_at','closed_at', 'repository', 'username', 'title', 'message', 'id','type']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Funtion to track rows
                def track_rows_in_csv(row_count, file_count, max_rows_per_file, fieldnames):
                    row_count += 1
                    if row_count >= max_rows_per_file:
                        csvfile.close()
                        file_count += 1
                        csvfile = open(os.path.join(self.output_dir, f"performance_data_{file_count}.csv"), 'a', newline='', encoding='utf-8')
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        # Reset row counter
                        row_count = 0

                # Funtion to write rows
                def write_rows(a,b,c,d,e,f,g,h):
                    writer.writerow({
                                    'created_at': a,
                                    'closed_at' : b,
                                    'repository': c,
                                    'username': d,
                                    'title' : e,
                                    'message': f,
                                    'id' : g,
                                    'type': h
                                })


                # Initialize row counter
                row_count = 0

                # Search for repositories matching the query
                results = g.search_repositories(self.query)
                counter = 1 # Track Quantity of data
                print(f"Total repositories found: {results.totalCount}")

                for repo in results:
                    print(f"Processing Repo no : {counter} ---- {repo.full_name}")

                    # Search for issues
                    try:
                        issues = repo.get_issues(state='all')
                        if issues is not None:
                            for issue in issues:
                                if issue is not None:
                                    if issue.title and issue.body is not None:
                                        if any(re.search(r'\b{}\w*'.format(keyword), f'{issue.title.lower()} {issue.body.lower()}') for keyword in self.keywords):
                                            if issue.state == "open":
                                                write_rows(issue.created_at, 
                                                        "", 
                                                        repo.full_name, 
                                                        issue.user.login, 
                                                        issue.title, 
                                                        issue.body, 
                                                        issue.id, 
                                                        issue.state + "_" + "issue")
                                                track_rows_in_csv(row_count, file_count, max_rows_per_file, fieldnames)
                                            else:
                                                write_rows(issue.created_at, issue.closed_at, repo.full_name, issue.user.login, issue.title, issue.body, issue.id, issue.state + "_" + "issue")
                                                track_rows_in_csv(row_count, file_count, max_rows_per_file, fieldnames)
                    except RateLimitExceededException as rate_limit_exception:
                        reset_time = rate_limit_exception.rate.reset.timestamp()
                        print(f"Rate limit exceeded. Waiting until {reset_time} to retry...")
                        time.sleep(reset_time - time.time())  # Wait until the rate limit is reset
                        continue
                    except Exception as e:
                        print(f"Exception occured: {e}")
                        time.sleep(90)
                        continue

                    # Search for pull requests
                    try:
                        pulls = repo.get_pulls(state="all")
                        if pulls is not None:
                            for pull in pulls:
                                if pull is not None:
                                    if pull.title and pull.body  is not None:
                                        if any(re.search(r'\b{}\w*'.format(keyword), f'{pull.title.lower()} {pull.body.lower()}') for keyword in self.keywords):
                                            if pull.state == "open":
                                                write_rows(pull.created_at, "", repo.full_name, pull.user.login, pull.title, pull.body, pull.id, pull.state + "_" + "pull_request")
                                                track_rows_in_csv(row_count, file_count, max_rows_per_file, fieldnames)
                                            else:
                                                write_rows(pull.created_at, pull.closed_at, repo.full_name, pull.user.login, pull.title, pull.body, pull.id, pull.state + "_" + "pull_request")
                                                track_rows_in_csv(row_count, file_count, max_rows_per_file, fieldnames)
                    except RateLimitExceededException as rate_limit_exception:
                        reset_time = rate_limit_exception.rate.reset.timestamp()
                        print(f"Rate limit exceeded. Waiting until {reset_time} to retry...")
                        time.sleep(reset_time - time.time())  # Wait until the rate limit is reset
                        continue
                    except Exception as e:
                        print(f"Exception occured: {e}")
                        time.sleep(90)
                        continue
                        
                    # Search for commits
                    try:
                        commits = repo.get_commits()
                        if commits is not None:
                            for commit in commits:
                                if commit is not None:
                                    if commit.author and commit.commit.message is not None:
                                        if any(re.search(r'\b{}\w*'.format(keyword), f'{commit.commit.message.lower()}') for keyword in self.keywords):
                                            write_rows(commit.commit.author.date ,commit.commit.author.date ,repo.full_name ,commit.commit.author.name ,commit.commit.message.split('\n')[0],
                                                    commit.commit.message, commit.sha, "commit")
                                            track_rows_in_csv(row_count, file_count, max_rows_per_file, fieldnames)
                    except RateLimitExceededException as rate_limit_exception:
                        reset_time = rate_limit_exception.rate.reset.timestamp()
                        print(f"Rate limit exceeded. Waiting until {reset_time} to retry...")
                        time.sleep(reset_time - time.time())  # Wait until the rate limit is reset
                        continue
                    except Exception as e:
                        print(f"Exception occured: {e}")
                        time.sleep(90)
                        continue
                    counter += 1
                    clear_output(self.output_text)
                    print("\n")
                total = (file_count * self.num_row_per_file) + row_count
                print(f"{total} data from {counter} repositories is mined and stored successfully")
                end = time.time()
                print(f"Total time elapsed is {end - start}")
                messagebox.showinfo("Data Mined Successfully", f"{total} data from {counter} repositories is mined and stored successfully")
                self.mine_button.config(state="normal")
            except Exception as e:
                print(f"Error Occured: {e}")
                self.mine_button.config(state="normal")
        threading.Thread(target=parallel_mining).start()

    def open_info(self):
        self.query_entry.insert(0, "language:java stars:>100")
        self.api_entry.insert(0, "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")
        self.output_entry.insert(0, "Output")
        self.row_entry.insert(0, int("50000"))
        messagebox.showinfo("Important", "These are just sample inputs\nYou can change the query as per your requirnment\nAlso please change the **API access token** (This one maybe expired)")


#################################################################
###################### Create Model #############################
class Screen2(tk.Toplevel):
    def __init__(self):
        super().__init__()
        
        self.csv_path = None
        self.message_column_name = ""
        self.label_column_name = ""
        self.selected_models = []
        self.data = None

        self.title("Performance Issue Classifier: Create Model")
        self.geometry("640x520")
        self.resizable(False, False)
        # Logo
        logo = tk.PhotoImage(file = resource_path("logo.png"))
        self.iconphoto(False, logo)

        _canvas = tk.Canvas(self, width=640, height=400, bd=0, borderwidth=0)
        _canvas.pack()

        # Background
        self.bg_img = tk.PhotoImage(file= resource_path("image.png"))
        _canvas.create_image(0,0, image= self.bg_img, anchor="nw")

        # Csv label and upload button
        _canvas.create_text(70,22, text="Select Labeled Dataset: Folder contain csv file/s", font=("Open Sans", 10), anchor="nw")
        self.csv_button = tk.Button(self, text="Select folder", command=self.select_csv)
        self.csv_button.place(x=420, y=20)
        # message Column Name label and text area
        _canvas.create_text(70,52, text="Type column name which contains messages:", font=("Open Sans", 10), anchor="nw")
        self.message_col_entry = tk.Entry(self)
        self.message_col_entry.place(x=420, y=55)
        # Label Column Name label and text area
        _canvas.create_text(70,82, text="Type column name which contains labels:", font=("Open Sans", 10), anchor="nw")
        self.label_col_entry = tk.Entry(self)
        self.label_col_entry.place(x=420, y=80)
        # Label for Models Selection
        _canvas.create_text(230,122, text="***** Select Models to train *****", font=("Open Sans", 10), anchor="nw")
        # Create IntVar variables to hold the state of checkboxes
        self.perceptron_var = tk.IntVar()
        self.multinomial_nb_var = tk.IntVar()
        self.svc_var = tk.IntVar()
        self.linear_reg_var = tk.IntVar()
        self.random_forest_var = tk.IntVar()
        # Create checkboxes for each model
        self.perceptron_checkbox = tk.Checkbutton(self, text="Perceptron", variable=self.perceptron_var)
        self.multinomial_nb_checkbox = tk.Checkbutton(self, text="Multinomial Naive Bayes", variable=self.multinomial_nb_var)
        self.svc_checkbox = tk.Checkbutton(self, text="SVC", variable=self.svc_var)
        self.linear_reg_checkbox = tk.Checkbutton(self, text="Linear Regression", variable=self.linear_reg_var)
        self.random_forest_checkbox = tk.Checkbutton(self, text="Random Forest", variable=self.random_forest_var)
        # Place the checkboxes on the application window
        self.perceptron_checkbox.place(x=30, y=150)
        self.multinomial_nb_checkbox.place(x=130, y=150)
        self.svc_checkbox.place(x=300, y=150)
        self.linear_reg_checkbox.place(x=370, y=150)
        self.random_forest_checkbox.place(x=500, y=150)
        # Button to Create Model
        create_button = tk.Button(self, text="Create Model", width=20 ,command=self.create_model, bg="#0085E2", fg="white")
        create_button.place(x=240, y=200)
        info_button = tk.Button(self, text="How to use", command=self.open_info) 
        info_button.place(x=20, y=200)
        # Text widget to display the print statements
        _canvas.create_text(0,243, text=statement, fill="#01364a", anchor="nw")
        self.output_text = tk.Text(self, bg="#01364a", fg="white")
        self.output_text.place(x=0, y=260, height=260)
        redirect_output(self.output_text)
        self.grab_set() #(block access to main window)

    # Helper Function to load a csv file
    def select_csv(self):
        self.csv_path = filedialog.askdirectory(title="Select File or Folder")
        print("CSV file:", self.csv_path)
        self.data = loaddata(self.csv_path)
        if "message_cleaned" in self.data.columns and "label" in self.data.columns:
            self.message_col_entry.insert(0, "message_cleaned")
            self.label_col_entry.insert(0, "label")
            print("System has automatically selected the message and label column")
        else:
            print("Please write the column names for messages and labels")
            print("Dataset's Columns are:")
            print(self.data.columns)
        print('#' * 80)

    # A funtion to plot the pie chart represents the distribution of the labels
    def plot_distribution(self, dataset, name):
        # If labels are single values
        if isinstance(dataset[0], str):
            type_counts = dataset.value_counts()
            # Plot a pie chart of the most common types of performance issues
            plt.figure(figsize=(10, 5))
            plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            plt.title("Distribution of the dataset")
            plt.tight_layout()
            plt.savefig(name + ".png")
        # If labels are lists
        elif isinstance(dataset[0], list):
            all_labels = [label for sublist in dataset for label in sublist]
            type_counts = np.unique(all_labels, return_counts=True)
            # Plot a pie chart of the most common types of performance issues
            plt.figure(figsize=(10, 5))
            plt.pie(type_counts[1], labels=type_counts[0], autopct='%1.1f%%')
            plt.title("Distribution of the dataset")
            plt.tight_layout()
            plt.savefig(name + ".png")

    # Helper Funtion to perform Random Oversampling on the dataset
    def balance_dataset(self, dataset, col_name):
        print("---------- Performing over-sampling")
        # Separate the data for each label
        label_data = {}
        balanced_data = []
        # Group the data by label
        for label in dataset[col_name].unique():
            label_data[label] = dataset[dataset[col_name] == label]
        # Determine the maximum number of samples among all labels
        max_samples = max(len(data) for data in label_data.values())
        # Oversample the data for each label to match the maximum number of samples
        for label, data in label_data.items():
            oversampled_data = resample(data, replace=True, n_samples=max_samples, random_state=42)
            balanced_data.append(oversampled_data)
        # Combine the balanced data for all labels into a single DataFrame
        balanced_df = pd.concat(balanced_data)
        # Reset the index of the balanced DataFrame
        balanced_df = balanced_df.reset_index(drop=True)
        # Shuffle the balanced DataFrame to remove any type of sequence
        balanced_df = shuffle(balanced_df)
        return balanced_df
    
    # Helper Funtion to select models
    def select_models(self):
        # Check which models were selected using the checkboxes
        if self.perceptron_var.get():
            self.selected_models.append(Perceptron(random_state=42))
        if self.multinomial_nb_var.get():
            self.selected_models.append(MultinomialNB())
        if self.svc_var.get():
            self.selected_models.append(SVC(random_state=42))
        if self.linear_reg_var.get():
            self.selected_models.append(LinearRegression())
        if self.random_forest_var.get():
            self.selected_models.append(RandomForestClassifier(random_state=42))
        return self.selected_models

    # The main function which will apply the vectorizer and train the model for furthur classification
    def create_model(self):
        try:
            def multi_thread():

                self.perceptron_checkbox.config(state="disabled")
                self.multinomial_nb_checkbox.config(state="disabled")
                self.svc_checkbox.config(state="disabled")
                self.linear_reg_checkbox.config(state="disabled")
                self.random_forest_checkbox.config(state="disabled")
                self.message_column_name = self.message_col_entry.get()
                self.label_column_name = self.label_col_entry.get()
                
                if self.csv_path is None:
                    messagebox.showerror("Error", "Please upload labeled dataset")
                    return
                if not self.message_column_name or not self.label_column_name:
                    messagebox.showerror("Error", "Please enter the Column Names for Messages and Labels")
                    return
                models = self.select_models()
                if models == []:
                    messagebox.showerror("Error", "Please select atleast one model")
                    return
                start = time.time()
                clear_output(self.output_text)
                print(f"Selected Models are: {self.selected_models}")
                # Perform Oversampling
                data = data.drop_duplicates(subset=self.message_column_name, keep="last")
                self.plot_distribution(self.data[self.label_column_name], "before")
                new_df = self.balance_dataset(self.data, self.label_column_name)
                self.plot_distribution(new_df[self.label_column_name], "after")
                # Split the data into training and test sets
                y = new_df[self.label_column_name].values
                X = new_df[self.message_column_name].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # Extract features from the text data
                print("---------- Applying TFIDF Vectorizer (Please be paitence it will take some time)")
                vectorizer = TfidfVectorizer()
                # Fit and transform the training data
                X_train_features = vectorizer.fit_transform(X_train)
                # Transform the test data
                X_test_features = vectorizer.transform(X_test)
                # Model Creation
                print("---------- Creating Model")
                best_model = None
                best_accuracy = 0.0
                for model in models:
                    # Fit the model and predict on the test set
                    model.fit(X_train_features, y_train)
                    y_pred = model.predict(X_test_features)
                    # Calculate and print the accuracy score
                    accuracy = accuracy_score(y_test, y_pred)
                    print(f'{model.__class__.__name__}: {accuracy:.3f}')
                    #print(classification_report(y_test, y_pred))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model

                print(f'\nBest model: {best_model.__class__.__name__} ({best_accuracy:.3f})')
                end = time.time()
                print(f"Total time elapsed is {end - start}")
                self.perceptron_checkbox.config(state="normal")
                self.multinomial_nb_checkbox.config(state="normal")
                self.svc_checkbox.config(state="normal")
                self.linear_reg_checkbox.config(state="normal")
                self.random_forest_checkbox.config(state="normal")
                return joblib.dump(best_model, 'best_model.pkl'), joblib.dump(vectorizer ,"vectorizer.pkl")
            threading.Thread(target=multi_thread).start()
        except Exception as e:
            self.perceptron_checkbox.config(state="normal")
            self.multinomial_nb_checkbox.config(state="normal")
            self.svc_checkbox.config(state="normal")
            self.linear_reg_checkbox.config(state="normal")
            self.random_forest_checkbox.config(state="normal")
            print(f"Error Occured: {e}")
        
    def open_info(self):
        messagebox.showinfo("How to Use", "1. Upload CSV file contains labeled dataset.\n2. Enter a column name which contains messages.\n3. Enter a column name which contain labels\n4. Select atleast one model to train.")


#################################################################
############### Perform Classification ##########################
class Screen3(tk.Toplevel):
    def __init__(self):
        super().__init__()

        self.model_path = None
        self.vectorizer_path = None
        self.csv_path = None
        self.column_name = "" 
        self.dataset = None

        # Add widgets specific to Screen 3 here
        self.title("Performance Issue Classifier: Perform Classification Window")
        self.geometry("640x520")
        self.resizable(False, False)
        # Logo
        logo = tk.PhotoImage(file = resource_path("logo.png"))
        self.iconphoto(False, logo)

        _canvas = tk.Canvas(self, width=640, height=400, bd=0, borderwidth=0)
        _canvas.pack()
        # Background
        self.bg_img = tk.PhotoImage(file= resource_path("image.png"))
        _canvas.create_image(0,0, image=self.bg_img, anchor="nw")
        # CSV Upload and Show
        _canvas.create_text(70,22, text="Select CSV file to classify:", font=("Open Sans", 10), anchor="nw")
        self.csv_button = tk.Button(self, text="Select CSV File", command=self.select_csv)
        self.csv_button.place(x=420, y=20)
        self.show_button = tk.Button(self, text="View", command=self.display_df_in_new_window, state="disabled")
        self.show_button.place(x=520, y=20)
        # Enter Message Column  
        _canvas.create_text(70,62, text="Type column name which contains messages:", font=("Open Sans", 10), anchor="nw")
        self.message_col_entry = tk.Entry(self)
        self.message_col_entry.place(x=420, y=60)
        # Model Upload Label and button
        _canvas.create_text(70,102, text="Select model file:", font=("Open Sans", 10), anchor="nw")
        self.model_button = tk.Button(self, text="Select model", command=self.select_model)
        self.model_button.place(x=420, y=100)
        # Vectorizer Upload Label and button
        _canvas.create_text(70,142, text="Select vectorizer file:", font=("Open Sans", 10), anchor="nw")
        self.vectorizer_button = tk.Button(self, text="Select vectorizer", command=self.select_vectorizer)
        self.vectorizer_button.place(x=420, y=140)

        # Classify Button
        self.classify_button = tk.Button(self, text="Classify", width=20 ,command=self.classification, bg="#0085E2", fg="white")
        self.classify_button.place(x=240, y=200)

        # Text widget to display the print statements
        _canvas.create_text(0,243, text=statement, fill="#01364a", anchor="nw")
        self.output_text = tk.Text(self, bg="#01364a", fg="white")
        self.output_text.place(x=0, y=260, height=260)

        self.info_button = tk.Button(self, text="How to use", command=self.open_info) 
        self.info_button.place(x=20, y=200)
        redirect_output(self.output_text)
        self.grab_set()


    # Helper Funtion to generate the classification report
    def rep_calculate(self, data):
        # Group by repository and label, and calculate counts
        grouped_df = data.groupby(["repository", "predicted_class"]).size().unstack(fill_value=0)
        # Add a "Total" column
        grouped_df["Total"] = grouped_df.sum(axis=1)
        # Display the resulting dataset
        grouped_df.to_csv(os.getcwd()+"/output/report.csv", header=True )

    # Helper Funtion to plot the top then repositories classification
    def plot_dis(self, data):
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

    # Helper Function to show data
    def display_df_in_new_window(self):
        frame = Toplevel(self) #this is the new window
        table = Table(frame, dataframe=self.dataset, showtoolbar=True, showstatusbar=True)
        table.show()

    # Helper Function to load a csv file
    def select_csv(self):
        self.csv_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
        print("CSV file:", self.csv_path)
        self.dataset = loaddata(self.csv_path)
        if "message" in self.dataset.columns:
            self.message_col_entry.insert(0, "message")
            print("System has automatically selected the message column")
        else:
            print("Dataset's Columns are:")
            print(self.dataset.columns)
        print('#' * 80)
        if self.csv_path is not None:
            self.show_button['state']="normal"

    # Helper Function to load a trained model
    def select_model(self):
        self.model_path = filedialog.askopenfilename(title="Select Model", filetypes=[("Pickle Files", "*.pkl")])
        print("\nModel file: ", self.model_path)

    # Helper Function to load a trained vectorizer
    def select_vectorizer(self):
        self.vectorizer_path = filedialog.askopenfilename(title="Select Vectorizer", filetypes=[("Pickle Files", "*.pkl")])
        print("Vectorizer file: ", self.vectorizer_path)
            
    # Helper function to extract java code fragments
    def extract_java_code(self, msg):
        java_code_fragments = re.findall(r'```java([\s\S]*?)```', str(msg))
        return java_code_fragments
    
    # Helper Function to check model, vectorizer and select column contains message 
    def classification(self):
        try:
            if self.model_path is None or self.vectorizer_path is None or self.csv_path is None:
                print("Please select all files")
                messagebox.showerror("Error Occured", "Please select csv, model and vectorizer (all) files")
                return
            def main_func():
                self.column_name = self.message_col_entry.get()
                if not self.column_name:
                    messagebox.showerror("Error", "Please enter the Column Name")
                    return
                # Create a new folder for outputs
                output_folder = "output"
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                #################################
                start = time.time()
                clear_output(self.output_text)
                self.classify_button.config(state = "disabled")
                df = self.dataset
                # df = df.drop_duplicates(subset=column_name)
                print("---------- Preparing Dataset")
                if self.column_name in df.columns:
                    df = processing.preprocessing(df, self.column_name)
                    df = processing.clean_stop_words(df, self.column_name)
                    print("---------- Classifing Messages")
                    model = joblib.load(self.model_path)
                    vectorizer = joblib.load(self.vectorizer_path)
                    messages = df[self.column_name+"_"+"cleaned"]
                    messages_features = vectorizer.transform(messages)
                    predicted_classes = model.predict(messages_features)
                    df["predicted_class"] = predicted_classes

                    if 'repository' in df.columns:
                        self.rep_calculate(df)
                        self.plot_dis(df)
                        
                    def extract_java_code(msg):
                        msg_code = re.findall(r'```java([\s\S]*?)```|```([\s\S]*?)```', str(msg))
                        return msg_code
                    
                    print("---------- Extracting Code Fragments")
                    df['java_code_fragment'] = df['message'].apply(extract_java_code)

                    print("\nClassification Completed Successfully")
                    df = df.drop(self.column_name+"_"+"cleaned", axis=1)
                    print("\nSaving File")
                    df.to_csv(os.getcwd()+"/output/classified data.csv", index=False, header=True)
                    end = time.time()
                    print(f"Total time elapsed is {end - start}")
                    self.classify_button.config(state = "normal")
                    messagebox.showinfo("Classification Completed!", \
                                        "Classification completed successfully\n \
                                        \nYou can find the files in 'Output' Folder")
                else:
                    print(f"\nThe -{self.column_name}- column not exist in the dataset")
                    self.classify_button.config(state = "normal")
                    messagebox.showerror("Error Occured", f"The -{self.column_name}- column not exist in the dataset")

            threading.Thread(target=main_func).start()        
        except Exception as e:
            print(f"Error Occured: {e}")

    # Show Info about this page
    def open_info(self):
        messagebox.showinfo("How to Use", "1. Select a csv file conatain messages to classify.\n2. Type a name of column in which messages are present in the csv file.\n3. Select a model file which is trained using the app.\n4. Select a vectorizer file which is trained using the app")


#################################################################
############# Run Main App (Initiate Main Screen) ###############
if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
