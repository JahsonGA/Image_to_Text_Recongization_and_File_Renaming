from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import re


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

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

def ExtractTEXT (filename,path,TextPath):
    img = cv2.imread(filename,0)        #opens image in grayscale
    
    #Apply threshold to create a binary image
    #ret,thresh = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
    #Normalize image
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    
    #No skew should be needed since the image is scanned in
    
    #Image Scaling
    # OCR preforms best when the image is at least 300 PPI
    img = set_image_dpi(filename)
    
    #Remove noise
    # removes the small dots/patches which have high intensity compared to the rest of the image for smoothening of the image. 
    img = remove_noise(img)
    
    #Thinning and Skeletonization
    # is performed for the handwritten text, as different writers use different stroke widths to write. This step makes the width of strokes uniform. 
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img, kernel, iterations = 1)

    # Perform OCR using pytesseract
    text = pytesseract.image_to_string(img)
    
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
    
    with open(os.path.join(TextPath,file),'w') as txt_file:                 #writes to text file the extracted image file. 
        txt_file.write(text)
        
    
   
if __name__ == "__main__":
    filename = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\20231207095006269.tif"
    path = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file"
    TextPath = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\Textfiles"
    ExtractTEXT(filename,path,TextPath)
    