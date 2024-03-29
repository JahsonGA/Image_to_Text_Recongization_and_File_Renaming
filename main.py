import cv2
import pytesseract
import numpy as np
import PIL
import nltk
import ExtractText
import FIndKeyData
#import pandas as pd

print(cv2.__version__)
print(pytesseract.__version__)
print(np.__version__)
print(PIL.__version__)
print(nltk.__version__)
#print(pd.__version__)

input = ".\\unnamed_file"
output = ".\\unnamed_file\\Textfiles"
ExtractText.extract_text_from_folder(input,output)
input_folder = ".\\unnamed_file\\Textfiles"
image_folder = ".\\unnamed_file"
output_folder = ".\\complete_images"
manual_review_folder = ".\\manual_review_images"
# Move files based on keywords
FIndKeyData.move_files(input_folder, output_folder, manual_review_folder, image_folder)
# read_text_file_and_rename_image(input_folder)
# read_text_file_and_rename_image(input_folder)
# read_text_file_and_rename_image(input_folder)
print("Made it to the end")