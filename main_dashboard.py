import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess
import os

# Path to the software directory
software_path = r"C:\Users\paulo\Desktop\clinoform\software"

# Functions to Launch Each Module
def launch_digitalizer():
    try:
        # Run Digitalizer_1.0.py
        subprocess.run(["python", os.path.join(software_path, "Digitalizer_1.0.py")], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Error launching Digitalizer: {e}")

def launch_clinoform_analysis():
    try:
        # Run Clinoform_Analysis_1.0.py
        subprocess.run(["python", os.path.join(software_path, "Clinoform_Analysis_1.0.py")], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Error launching Clinoform Analysis: {e}")

def launch_profile_extraction():
    try:
        # Run Profile_Extractor_1.0.py
        subprocess.run(["python", os.path.join(software_path, "Profile_Extractor_1.0.py")], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Error launching Profile Extraction: {e}")

# Main Dashboard Class
class MainDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Main Dashboard")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # Create a menu bar
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # Add 'File' menu
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # Add 'Help' menu
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

        # Welcome Label
        self.welcome_label = tk.Label(self.root, text="Welcome to the Analysis Toolkit", font=("Arial", 18))
        self.welcome_label.pack(pady=20)

        # Button Frame
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=20)

        # Add Buttons
        self.digitalizer_button = ttk.Button(self.button_frame, text="Digitalizer", command=launch_digitalizer)
        self.digitalizer_button.grid(row=0, column=0, padx=20, pady=20)

        self.clinoform_button = ttk.Button(self.button_frame, text="Clinoform Analysis", command=launch_clinoform_analysis)
        self.clinoform_button.grid(row=0, column=1, padx=20, pady=20)

        self.profile_button = ttk.Button(self.button_frame, text="Profile Extraction", command=launch_profile_extraction)
        self.profile_button.grid(row=0, column=2, padx=20, pady=20)

        # Footer
        self.footer_label = tk.Label(self.root, text="Select an application to proceed.", font=("Arial", 12), fg="gray")
        self.footer_label.pack(side=tk.BOTTOM, pady=20)

    # Show About Info
    def show_about(self):
        messagebox.showinfo("About", "This is the Analysis Toolkit version 1.0\nDeveloped for analysis and research purposes.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainDashboard(root)
    root.mainloop()
