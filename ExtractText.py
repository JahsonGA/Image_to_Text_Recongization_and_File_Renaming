from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import re


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = temp_file.NamedTemporaryFile(delete=False, suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def preprocess_image(image):
    # Convert to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Ensure the image is of type uint8
    gray = np.uint8(gray)
    
    # Apply adaptive thresholding
    #thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    
    # Invert the image
    inverted = cv2.bitwise_not(image)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoisingColored(gray, None, 10, 10, 7, 15)
    
    return denoised

    '''TODO 
    Binary conversion of color/gray scale images using PIL/Opencv.
    Remove pictures from image as contours with largest area compared to average area of all the contours present in the image.
    Remove lines using canny edge filter and houghlines
    Use RLSA(run length smoothing algorithm) on this binary image. '''

def ExtractTEXT(filename, path, TextPath):
    img = cv2.imread(filename,-1)  # Open image in without changing color
    
    if np.shape(img) == ():
        print("Error")
    
    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(preprocessed_img)

    # Display the text
    print("Text Extracted:")
    print(text)

    # Find bounding boxes around the words
    boxes = pytesseract.image_to_boxes(preprocessed_img)
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(preprocessed_img, (int(b[1]), img.shape[0] - int(b[2])), (int(b[3]), img.shape[0] - int(b[4])), (255, 255, 255), 2)

    # Show the image with bounding boxes
    preprocessed_img = cv2.resize(preprocessed_img, (800, 900))  # Resize image for display
    cv2.imshow('Image with Bounding Boxes', preprocessed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Write the extracted text to a text file
    file = os.path.basename(re.sub('.tif+', '', filename) + ".txt")  # Find the file name for the text file using regex

    with open(os.path.join(TextPath, file), 'w') as txt_file:  # Write to the text file the extracted image file
        txt_file.write(text)

        
    
   
if __name__ == "__main__":
    filename = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\20231207095006269.tif"
    path = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file"
    TextPath = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\Textfiles"
    ExtractTEXT(filename,path,TextPath)