from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import re
import tempfile


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.LANCZOS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename

def preprocess_image(image):
    # Convert to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Binary conversion
    x, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Remove small contours
    #   findContours returns a modified image, contours of that image and the heirarchery 
    #   finds the bounds of an image
    contours, heir = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    avg_area = np.mean(areas)
    for cnt in contours:
        if cv2.contourArea(cnt) < avg_area:
            cv2.drawContours(binary, [cnt], -1, (0, 0, 0), cv2.FILLED)
    
    # Remove lines using canny edge filter and houghlines
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(binary, (x1, y1), (x2, y2), (0, 0, 0), 5)
    
    # Run Length Smoothing Algorithm (RLSA)
    kernel = np.ones((1, 20), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Invert the image
    binary = cv2.bitwise_not(binary)
    
    # Resize image for display
    cv2.imshow('Image with Bounding Boxes', cv2.resize(binary, (800, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    boxes = pytesseract.image_to_boxes(binary)
    
    for b in boxes.splitlines():
        b = b.split(' ')
        image = cv2.rectangle(binary, (int(b[1]), image.shape[0] - int(b[2])), (int(b[3]), image.shape[0] - int(b[4])), (255, 255, 255), 2)

    
    return binary

def ocr_from_image(image):
    # Set image DPI
    image_with_dpi = set_image_dpi(image)
    
    # Preprocess the image
    image_with_dpi = cv2.imread(image_with_dpi, 0) # Open image in gray scale
    
    if np.shape(image_with_dpi) == ():
        print("Error: with opening image in gray scale")
    
    preprocessed_img = preprocess_image(image_with_dpi)
    
    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(preprocessed_img)
    
    return text

def ExtractTEXT(filename, path, TextPath):
    text = ocr_from_image(filename)  
    
    # Display the text
    print("Text Extracted:")
    print(text)

    

    # Write the extracted text to a text file
    file = os.path.basename(re.sub('.tif+', '', filename) + ".txt")  # Find the file name for the text file using regex

    with open(os.path.join(TextPath, file), 'w') as txt_file:   # Write to the text file the extracted image file
        txt_file.write(text)                                    # ding boxes around the words        
    
   
if __name__ == "__main__":
    filename = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\20231207095006269.tif"
    path = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file"
    TextPath = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\Textfiles"
    ExtractTEXT(filename,path,TextPath)