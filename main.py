import cv2
import pytesseract
import numpy as np
import PIL
import nltk
import ExtractText
import FindKeyData
import FileOrg
#import pandas as pd

print(cv2.__version__)
print(pytesseract.__version__)
print(np.__version__)
print(PIL.__version__)
print(nltk.__version__)
#print(pd.__version__)

input = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file"
output = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTAP history\\unnamed_file\\Textfiles"
ExtractText.extract_text_from_folder(input,output)
print("Extracted text")

input_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTAP history\\unnamed_file\\Textfiles"
image_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTAP history\\unnamed_file"
output_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTAP history\\unnamed_file\\complete_images"
manual_review_folder = "PennTAP history\\unnamed_file\\manual_review_images"

# Move files based on keywords
FindKeyData.move_files(input_folder, output_folder, manual_review_folder, image_folder)
print("Find Key Data")

text_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTAP history\\unnamed_file\\Textfiles"
image_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTAP history\\unnamed_file\\complete_images"
folder_path = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTAP_History_1973-1975.csv"

# Initialize the CSV file with headers (run only once)
FileOrg.initialize_csv(folder_path)

# Add file details
FileOrg.add_file_details(image_folder, text_folder, folder_path)
print("Files Moved")

print("Made it to the end")