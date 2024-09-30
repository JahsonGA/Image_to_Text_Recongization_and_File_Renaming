# Creates csv file, and writes file, date uploaded, date of the article, description, and tags
import csv
from datetime import datetime
import os

# Function to append file data to CSV
def write_to_csv(file_name, article_date, description, tags, csv_file='file_metadata.csv'):
    # Get the current date as the upload date
    date_uploaded = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open the CSV file in append mode ('a')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the row to the CSV file
        writer.writerow([file_name, date_uploaded, article_date, description, tags])
    
    print(f"File '{file_name}' has been added to {csv_file}.")

# Function to prompt user for input and add the file details to the CSV
def add_file_details(image_folder,textfolder):
    try:
        for file in os.listdir(image_folder):
            file_name = file
            article_date = file[:8]
            
            
            

        # Prompt for details (once for all files)
        description = input("Enter a brief description for all files: ")
        tags = input("Enter tags (comma separated) for all files: ")
        
        # Write each file's details to the CSV
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            write_to_csv(file_path, article_date, description, tags)
    
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")

# Ensure the CSV has a header if the file is newly created
def initialize_csv(csv_file='file_metadata.csv'):
    try:
        # Check if file exists, if not create with headers
        with open(csv_file, mode='r') as file:
            pass
    except FileNotFoundError:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["File Name", "Date Uploaded", "Date of Article", "Description", "Tags"])
        print(f"Created new CSV file '{csv_file}' with headers.")

# Initialize the CSV file with headers (run only once)
initialize_csv()

# Add file details
add_file_details()
if __name__ == "__main__":
    text_folder = ".\\unnamed_file\\Textfiles"
    image_folder = ".\\unnamed_file"