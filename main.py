import cv2
import pytesseract
import numpy as np
import PIL
import nltk
import ExtractText
import FindKeyData
import FileOrg

print(cv2.__version__)
print(pytesseract.__version__)
print(np.__version__)
print(PIL.__version__)
print(nltk.__version__)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#
#
#   Make the changes to the five lines below
#
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
unnamed_files = "C:\\Users\\jfg5801\\Desktop\\unnamed_file"
text_files = "C:\\Users\\jfg5801\\Desktop\\unnamed_file\\Textfiles"
compled_files = "C:\\Users\\jfg5801\\Desktop\\complete_images"
manual_review_folder = "C:\\Users\\jfg5801\\Desktop\\manual_review_images"
CSV_path = "C:\\Users\\jfg5801\\Desktop\\PennTAP_History_1971_1973.csv"
 
# Extract text from images
ExtractText.extract_text_from_folder(unnamed_files,text_files)
print("Extracted Text")
 
# Move files based on keywords
FindKeyData.move_files(text_files, compled_files, manual_review_folder, unnamed_files)
print("Files Moved")
 
# Initialize the CSV file with headers (run only once)
FileOrg.initialize_csv(CSV_path)
 
# Add file details
FileOrg.add_file_details(compled_files, text_files, CSV_path)
 
print("Made it to the end")