import pytesseract
import numpy as np
import cv2
import os
import tempfile

# ML
from imutils.object_detection import non_max_suppression
import argparse
import time

# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

model = os.getenv("model")
path = os.getenv("path")

#! BEFORE made public hide file path!
#TODO hide true path
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

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

'''counts white pixels'''
def get_white_pixel_count(image):
    return np.sum(image == 255)

#Helper function for determining which image processing algorithm to use
def needs_noise_reduction(image):
    # Calculate the average pixel intensity of the image
    avg_intensity = cv2.mean(image)[0]
    # If the average intensity is below a threshold, consider it noisy
    return avg_intensity < 100

def apply_noise_reduction(image):
    # Apply Gaussian blur for noise reduction
    return cv2.GaussianBlur(image, (5, 5), 0)

def needs_contrast_enhancement(image):
    # Calculate the histogram of the image
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])
    # Check if the histogram is mostly concentrated in the lower intensities
    return np.argmax(hist) < 50 or np.argmax(hist) > 200

def apply_contrast_enhancement(image):
    # Apply histogram equalization for contrast enhancement
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray_image)

def needs_image_resizing(image):
    # Check if the image dimensions exceed a certain threshold
    return image.shape[0] > 1000 or image.shape[1] > 1000

def apply_image_resizing(image):
    # Resize the image to a maximum of 1000x1000
    return cv2.resize(image, (1000, 1000))

def needs_color_correction(image):
    # Calculate the mean pixel intensity of the image
    mean_intensity = cv2.mean(image)[0]
    # If the mean intensity is below a threshold, consider it dark and needing color correction
    return mean_intensity < 100

def apply_color_correction(image):
    # Convert the image to grayscale for color correction
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def needs_segmentation(image):
    # Check if the image has large areas of uniform color
    _, std_dev = cv2.meanStdDev(image)
    return std_dev < 10

def apply_segmentation(image):
    # Apply thresholding for segmentation
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

def needs_feature_extraction(image):
    # Check if the image has complex textures or patterns
    return cv2.Laplacian(image, cv2.CV_64F).var() > 100

def extract_features(image):
    # Extract edges using Canny edge detection for feature extraction
    return cv2.Canny(image, 100, 200)

'''Shows the edited image with boxes will the letter opencv sees'''

def show_detected_text(image_path):
    # Read the image from preprocessed image
    image = cv2.imread(image_path)
    # Perform text detection using pytesseract
    boxes = pytesseract.image_to_boxes(image_path)        
        
        # Draw bounding boxes and write text in red
    for box in boxes.splitlines():
        box = box.split(' ')
        x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        image = cv2.rectangle(image, (int(box[1]), image.shape[0] - int(box[2])), (int(box[3]), image.shape[0] - int(box[4])), (0, 0, 255), 2) # Red color for bounding box
        cv2.putText(image, box[0], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA) # Red color for text
    
    # Display the image
    cv2.imshow('Detected Text', cv2.resize(image,(700,800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''Shows the edited image with boxes will the letter opencv sees when given a image object rather than a path string'''    

def show_detected_text_from_image(image):
    # Read the image from preprocessed image
    # image = preprocessing(image)
    # Perform text detection using pytesseract
    boxes = pytesseract.image_to_boxes(image)        
        
        # Draw bounding boxes and write text in red
    for box in boxes.splitlines():
        box = box.split(' ')
        x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
        image = cv2.rectangle(image, (int(box[1]), image.shape[0] - int(box[2])), (int(box[3]), image.shape[0] - int(box[4])), (0, 0, 255), 2) # Red color for bounding box
        cv2.putText(image, box[0], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) # Red color for text
    
    # Display the image
    cv2.imshow('Detected Text', cv2.resize(image,(700,800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def text_detection(image_path):
    img = preprocessing(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    save_text = pytesseract.image_to_data(rgb, output_type = pytesseract.Output.DICT)

    for i in range(0, len(save_text["text"])):
        x = save_text["left"][i]
        y = save_text["top"][i]
        w = save_text["width"][i]
        h = save_text["height"][i]

        text = save_text["text"][i]
        confidence_level = int(save_text["conf"][i])

        if confidence_level > 75:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y), (255, 255, 255), -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            # print(f"Confidence: {confidence_level}")
            # print(f"Text: {text}\n")

    # Display the image
    # cv2.imshow('Detected Text', cv2.resize(img,(700,850)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def preprocess_for_ocr(image_path):
    #TODO refine image processing
    # Read the image
    image = cv2.imread(image_path)

    # Increase Contrast
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)

    #TODO
    # 1) Find a way to calculate the number of binary and inverted binary pixel and pick the value that is 'better'
    # 2) Look for a way to correct skew in sections of an image.
    #   2a) This might not be possible within time frame.
    # Once this is complete the text detection and extraction should be done. 
    #*Compared to online  
    #   a) text detection my compares well but can still be improved when black on white background
    #   b) When inverted binary online text detection fails. which means mine is better
    #   c) Also fails for non standard text placement. Which means mine is the similar
    
    # Calculate white pixel count for binary and inverted binary images
    _, binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.THRESH_BINARY)
    _, inverted_binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.THRESH_BINARY_INV)
    white_pixel_count_binary = get_white_pixel_count(binary_image)
    white_pixel_count_inverted_binary = get_white_pixel_count(inverted_binary_image)

    # Choose the 'better' image based on white pixel count
    if white_pixel_count_binary >= white_pixel_count_inverted_binary:
        final_image = binary_image
    else:
        final_image = inverted_binary_image
    
    #! Rotates the whole image clockwise by 90 degrees instead of regions of an image. 
    '''# Used to rotate part of an image. these regions are correct to increase text recondition. 
    # Find contours and bounding boxes of text regions
    contours, _ = cv2.findContours(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the text region from the image
        text_region = final_image[y:y+h, x:x+w]

        # Apply skew correction to the text region
        deskewed_text_region = deskew(text_region)

        # Replace the text region in the original image with the deskewed region
        final_image[y:y+h, x:x+w] = deskewed_text_region'''
    
    #This section works the best of find both the most amount of words and the numbers with the least unknown values
    # global thresholding
    # ret1,binary_imageG = cv2.threshold(final_image,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    # ret2,binary_imageO = cv2.threshold(binary_imageG,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)'''
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(final_image,(5,5),0)
    ret3,binary_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Remove Small Objects
    kernel = np.ones((1, 1), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Enhance Text Regions
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)

    # Normalization (Optional)
    # norm_img = np.zeros((binary_image.shape[0], binary_image.shape[1]))
    # normalized_image = cv2.normalize(binary_image, norm_img, 0, 255, cv2.NORM_MINMAX)
    
    # Find contours and bounding boxes
    '''contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]'''
    
    # Removes pixels on object boundaries.
    erosion = cv2.erode(binary_image, kernel, iterations=1)

    # Image Scaling (DPI Adjustment)
    dpi_adjusted_image = set_image_dpi(erosion)
      
    return dpi_adjusted_image

def preprocessing(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Increase Contrast
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    #* Denoise increases run time exponentially for no increase from the 61% success rate of a sample of 92
    # denoised_image = cv2.fastNlMeansDenoising(enhanced_image, None, 20, 7, 21)
    # edge = cv2.Canny(enhanced_image,100,200)

    # Calculate threshold value for binary images transformation with Adaptive Thresh Gaussian Constant
    # This will allow a threshold to be calculated at every pixel matrix used.
    _, binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    '''
    The follow methods create a success rate of 61%
    #TODO try other methods and masks to try and improve performance
    ADAPTIVE_THRESH_GAUSSIAN_C
    THRESH_BINARY_INV
    '''
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(binary_image,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convex Hull Mask
    contours, _ = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hull_mask = np.zeros_like(th3)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(convex_hull_mask, [hull], -1, (255), -1)

    # Dilation of text found
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(convex_hull_mask, kernel, iterations = 1)

    # Bitwise AND with the original image
    # Creates mask
    characters_image = cv2.bitwise_and(binary_image, binary_image, mask=dilated_mask)
    '''
    NOTES
    ADAPTIVE_THRESH_GAUSSIAN_C threshold and binary_image mask result in 35/59
    ADAPTIVE_THRESH_GAUSSIAN_C threshold and th3 mask result in 33/59
    Thresh_binary_INV
    '''
    
    return characters_image

def preprocessing2(image_path):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
                    help="path to input image")
    ap.add_argument("-east", "--east", type=str,
                    help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
                    help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
                    help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

    # load the input image and grab the image dimensions
    image = cv2.imread(args["image"])
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    return 

#! This function is not complete will not work
def choose_preprocessing(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Increase Contrast
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # enhanced_image = cv2.equalizeHist(gray_image)

    # Calculate white pixel count for binary and inverted binary images
    # _, binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.THRESH_BINARY)
    # _, inverted_binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.THRESH_BINARY_INV)
    # white_pixel_count_binary = cv2.countNonZero(binary_image)
    # white_pixel_count_inverted_binary = cv2.countNonZero(inverted_binary_image)

    # Choose the preprocessing function based on white pixel count
    #* Will always use the preprocess_for_ocr()
    #if white_pixel_count_binary >= white_pixel_count_inverted_binary:
    #    # Use the preprocess_for_ocr function
    #    return preprocess_for_ocr(image_path)
    #else:
    #    # Use the preprocessing function
    #    return preprocessing(image_path)
    # Check if the image needs Noise Reduction
    if needs_noise_reduction(image):
        image = apply_noise_reduction(image)

    # Check if the image needs Contrast Enhancement
    if needs_contrast_enhancement(image):
        image = apply_contrast_enhancement(image)

    # Check if the image needs Image Resizing
    if needs_image_resizing(image):
        image = apply_image_resizing(image)

    # Check if the image needs Color Correction
    if needs_color_correction(image):
        image = apply_color_correction(image)

    # Check if the image needs Segmentation
    if needs_segmentation(image):
        image = apply_segmentation(image)

    # Check if the image needs feature extraction
    if needs_feature_extraction(image):
        features = extract_features(image)
        return features

    return image

# Extract text found in the image and write to a text file
def extract_text_from_folder(input, output):
    # Iterate over all files in the image folder
    for file_name in os.listdir(input):
        # Check if the file is a TIFF image
        if file_name.endswith(".tif"):
            # Construct the full path to the image file
            image_path = os.path.join(input, file_name)    
            # Extract text from the image
            # Perform OCR using pytesseract
            # img = preprocess_for_ocr(image_path)
            img = preprocessing(image_path)
            # img = preprocessing2(image_path)
            text = pytesseract.image_to_string(img)
            
            #TODO update new preprocessing2()
            
            # Calls display funcation
            # show_detected_text(img)
            # show_detected_text_from_image(img)
            
            # Construct the full path to the text file
            text_file_name = os.path.splitext(file_name)[0] + ".txt"
            text_file_path = os.path.join(output, text_file_name)
            # Write the extracted text to the text file
            with open(text_file_path, 'w') as text_file:
                text_file.write(text)
                
if __name__ == "__main__":
    input = ".\\unnamed_file"
    output = ".\\unnamed_file\\Textfiles"
    extract_text_from_folder(input,output)
    
#TODO create a readme to update everything that has been done. Send to Boss and list todos for when I get back. 
#TODO current idea is to use EAST ML method to find text in an image and then send the information found to a text file.