from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import re


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def ExtractTEXT (filename,path,TextPath):
    img = cv2.imread(filename,0)        #opens image in grayscale

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(img)
    
    #set location of text file.
    

    # Display the text
    print("Text Extracted:")
    print(text)

    # Find bounding boxes around the words
    boxes = pytesseract.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(img, (int(b[1]), img.shape[0] - int(b[2])), (int(b[3]), img.shape[0] - int(b[4])), (0, 255, 0), 2)

    # Show the image with bounding boxes
    img = cv2.resize(img, (800, 900))                                       #resize image
    cv2.imshow('Image with Bounding Boxes', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Write the extracted text to a text file
    file = os.path.basename(re.sub('.tif+','', filename) + ".txt")          #finds the file name for text file. using is the regex expression
    #file = os.path.splitext(file_location_name)                            #splitext to set the file to the name of same image file. 
    
    with open(os.path.join(TextPath,file),'w') as txt_file:                 #writes to text file the extracted image file. 
        txt_file.write(text)
        
    
   
if __name__ == "__main__":
    filename = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\20231207095006269.tif"
    path = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file"
    TextPath = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\Textfiles"
    ExtractTEXT(filename,path,TextPath)
    