import os
import sys
import processing
import tkinter as tk

# Helper function to take Relative path and join base path
def resource_path(relative_path):
    # Get absolute path to resource
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Helper function to print statements on console
def redirect_output(text_widget):
    class StdoutRedirector:
        def __init__(self, widget):
            self.widget = widget
        def write(self, message):
            self.widget.insert(tk.END, message)
            self.widget.see(tk.END)  # Automatically scroll to the end
            self.widget.update_idletasks()  # Update the widget immediately
        def flush(self):
            pass  # No need to implement flush for this redirector
    sys.stdout = StdoutRedirector(text_widget)

# Helper function to clear statements from console
def clear_output(text_widget):
    text_widget.delete("1.0", tk.END)

# Helper Function to load a dataset
def loaddata(csv_path):
    df = processing.read_csv_files(csv_path)
    return df