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
unnamed_files = "YourPathHere\\unnamed_file"
text_files = "YourPathHere\\unnamed_file"
compled_files = "YourPathHere\\unnamed_file"
manual_review_folder = "YourPathHere\\unnamed_file"
CSV_path = "YourPathHere\\unnamed_file"
 
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
 
print("\n\nMade it to the end")