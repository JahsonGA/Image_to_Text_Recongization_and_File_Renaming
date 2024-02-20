from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import re
import tempfile

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

'''Skew Correction
While scanning or taking a picture of any document, 
it is possible that the scanned or captured image might be slightly skewed sometimes. 
For the better performance of the OCR, it is good to determine the skewness in image and correct it.'''
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
    borderMode=cv2.BORDER_REPLICATE)
    return rotated

'''Image Scaling
To achieve a better performance of OCR, 
the image should have more than 300 PPI (pixel per inch). 
So, if the image size is less than 300 PPI, we need to increase it. 
We can use the Pillow library for this.'''
def set_image_dpi(image):
    length_x, width_y = image.shape[1], image.shape[0]
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename, im_resized)
    return temp_filename

'''Noise Removal
This step removes the small dots/patches 
which have high intensity compared to the rest of the image for smoothening of 
the image. OpenCV's fast Nl Means Denoising Coloured function can do that easily.'''
def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

'''Gray Scale image
This process converts an image from other color spaces to shades of Gray. 
The colour varies between complete black and complete white. 
OpenCV's cvtColor() function perform this task very easily.'''
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''Thresholding or Binarization
This step converts any colored image into a binary image that contains only two colors black and white. 
It is done by fixing a threshold (normally half of the pixel range 0-255, i.e., 127). 
The pixel value having greater than the threshold is converted into a white pixel else into a black pixel. 
To determine the threshold value according to the image Otsu's Binarization and Adaptive Binarization can be a better choice. 
In OpenCV, this can be done as given.'''
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]

def preprocess_for_ocr(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Skew Correction
    #deskewed_image = deskew(image)
    
    # Noise Removal
    denoised_image = remove_noise(image)

    # Grayscale Conversion
    grayscale_image = get_grayscale(denoised_image)

    # Thresholding
    thresholded_image = thresholding(grayscale_image)

    # Thinning and Skeletonization
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresholded_image, kernel, iterations = 1)

    # Normalization
    norm_img = np.zeros((erosion.shape[0], erosion.shape[1]))
    normalized_image = cv2.normalize(erosion, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Image Scaling
    dpi_adjusted_image = set_image_dpi(normalized_image)

    return dpi_adjusted_image

    #TODO TEST this on 2/22/24
    '''# Read the image
    image = cv2.imread(image_path)

    # Increase Contrast
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)

    # Thresholding
    _, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove Small Objects
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Enhance Text Regions
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Normalization (Optional)
    # norm_img = np.zeros((binary_image.shape[0], binary_image.shape[1]))
    # normalized_image = cv2.normalize(binary_image, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Image Scaling (DPI Adjustment)
    dpi_adjusted_image = set_image_dpi(binary_image)'''

def extract_text_from_folder(input, output):
     # Iterate over all files in the image folder
    for file_name in os.listdir(input):
        # Check if the file is a TIFF image
        if file_name.endswith(".tif"):
            # Construct the full path to the image file
            image_path = os.path.join(input, file_name)
            # Extract text from the image
            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(preprocess_for_ocr(image_path))
            # Construct the full path to the text file
            text_file_name = os.path.splitext(file_name)[0] + ".txt"
            text_file_path = os.path.join(output, text_file_name)
            # Write the extracted text to the text file
            with open(text_file_path, 'w') as text_file:
                text_file.write(text) 
    
   
if __name__ == "__main__":
    filename = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\20231207095006269.tif"
    input = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file"
    output = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\Textfiles"
    extract_text_from_folder(input,output)