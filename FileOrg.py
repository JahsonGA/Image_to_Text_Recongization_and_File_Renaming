# Creates csv file, and writes file, date uploaded, date of the article, description, and tags
import csv
from datetime import datetime
import os
import FindKeyData
import ExtractText
from pytesseract import image_to_string

# Function to append file data to CSV
def write_to_csv(file_name, article_date, description, tags, csv_file='PennTAP_History_1971-1973.csv'):
    # Get the current date as the upload date
    date_uploaded = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Open the CSV file in append mode ('a')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the row to the CSV file
        writer.writerow([file_name, date_uploaded, article_date, description, tags])
    
# Function to prompt user for input and add the file details to the CSV
def add_file_details(image_folder,textfolder,folder_path):
    try:
        for file in os.listdir(image_folder):
            file_name = file
            article_date = file[:10] 
            
            if(not article_date[0].isdigit()):
                article_date = "NoDateFound"

            # Prompt for details (once for all files)
            tags = file[11:] if len(file) > 8 else "No Tags"    #if not enough tags exist make it as so
            tags = tags.replace(".tif", "")
            
            # Process multi-page TIFF
            images = ExtractText.process_tiff_pages(image_folder)
            if not images:
                print(f"No pages found in {image_folder}. Skipping.")
                continue
            
            # Concatenate text from all pages
            description = ""
            for i in enumerate(images):
                preprocessed_image = ExtractText.preprocess_multi_page_tiff(image_folder)[i]
                page_text = image_to_string(preprocessed_image)
                description += f"\nPage {i + 1}:\n{page_text.strip()}\n"
        
            # Write each file's details to the CSV
            file_path = os.path.join(folder_path, file_name)
            
            # Get data about file description
            _, txt_file, text = FindKeyData.read_text_file_and_rename_image(textfolder)
            description = text.replace(",","").replace("\n","")  #get data from text detection
            
            # Remove the text file after processing
            txt_file_path = os.path.normpath(os.path.join(textfolder, txt_file))
            if os.path.isfile(txt_file_path):
                try:
                    os.remove(txt_file_path)
                except PermissionError as e:
                    print(f"Permission error: {e}. Skipping file: {txt_file_path}")
                except Exception as e:
                    print(f"Error: {e}. Skipping file: {txt_file_path}")
            else:
                print(f"Not a file: {txt_file_path}")
                
            write_to_csv(file_path, article_date, description, tags)
    
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found")

# Ensure the CSV has a header if the file is newly created
def initialize_csv(csv_file='PennTAP_History_1971-1973.csv'):
    try:
        # Check if file exists, if not create with headers
        with open(csv_file, mode='r') as file:
            pass
    except FileNotFoundError:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["File Name", "Date Uploaded", "Date of Article", "Description", "Tags"])

if __name__ == "__main__":
    text_folder = ".\\unnamed_file\\Textfiles"
    image_folder = ".\\complete_images"
    folder_path = "PennTAP_History_1971-1973.csv"
    
    # Initialize the CSV file with headers (run only once)
    initialize_csv(folder_path)

    # Add file details
    add_file_details(image_folder, text_folder, folder_path)