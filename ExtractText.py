import pytesseract
import numpy as np
import cv2
import os
from PIL import Image, TiffImagePlugin, UnidentifiedImageError

# ML
from imutils.object_detection import non_max_suppression

# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

model = os.getenv("model")
path = os.getenv("path")
package = os.getenv("package")

"""
#
#           Replace what is between ' with the location of the tesseract.exe file
#
"""
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Extract edges using Canny edge detection for feature extraction
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
    cv2.imshow(os.path.basename(image_path), cv2.resize(image,(700,800)))
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
    img =remove_noise_and_smooth(image_path)
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
def remove_noise_and_smooth(image):
    # Increase Contrast
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #adaptive threshold to filter out noise and enhance text visibility                             blocksize,constant C
    #   average threshold is subtracted by this constant. The hight the values help keep the average when images have varying lighting
    filter = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,41)
    #* changed ADAPTIVE_THRESH_MEAN_C to ADAPTIVE_THRESH_GAUSSIAN_C
            #? results in a 37/55 which is a 40.2% success rate
            #! increased block size from 9 to 11, results in 39.1.
    
    #kernel for morphological ops: erosion and dilation
    #   creates a matrix. used to smooth out an image. the large the matrix more smoothing done
    kernel = np.ones((1,1), np.uint8)
    #* changed matrix size to 2x2 from 1x1
        #! results in 15.2%
    #* changed matrix size to 3x3
        #! 1/91
    
    # Perform morphological opening to remove small noise regions
    opening = cv2.morphologyEx(filter, cv2.MORPH_OPEN, kernel)
    
    # perform morpthological closing to fill gaps in text regions
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    img = preprocess_for_ocr(image)
    
    or_img = cv2.bitwise_or(img,closing)
    
    return or_img
    
def preprocess_for_ocr(image):
    #smoothing image
    # image = cv2.imread(image_path) #needed if imread() was not called yet
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #binary thresholding
    #pixels with intensity greater than or equal to 88 are set to white while other are set to black
    ret1, th1 = cv2.threshold(image,88,255,cv2.THRESH_BINARY)
    #* increase divide between white and black pixel to 78 from 88
    #!  results of 40.2% 37/55
    #* increase divide between white and black pixel to 98 from 88
    #!  results of 40.2% 37/55
    
    #OTSU's Thresholding
    ret2, th2 = cv2.threshold(th1,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #gaussian blurring to reduce noise
    blur = cv2.GaussianBlur(th2,(9,9),0)
    #* increased matrix to 9x9 from 5x5
    #?  results in 42.68% 35/57
    #* increased matrix to 11x11 from 9x9
    #!  results in 40.2% 37/55
    #* increased matrix to 10x10 from 9x9
    #!  failed to complie, must be odd matrix
    
    #OTSU's Thresholding
    ret3, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #returns smoothed image
    return th3

def process_tiff_pages(image_path):
    images = []
    try:
        with Image.open(image_path) as img:
            for page in range(img.n_frames):  # Iterate through pages
                img.seek(page)
                if img.mode != 'RGB':  # Ensure the image is in RGB mode
                    img = img.convert('RGB')
                # Convert to OpenCV format for further processing
                cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                images.append(cv_image)
    except PermissionError as e:
        print(f"Permission denied: {image_path} - {e}\n\n If the file is open, close it and try again.")
    except FileNotFoundError as e:
        print(f"Error reading TIFF: {image_path} - {e}")
    except UnidentifiedImageError as e:  # Handle errors specific to PIL
        print(f"Error reading TIFF: {image_path} - {e}")
    except Exception as e:  # Handle unexpected errors
        print(f"Unexpected error with {image_path}\n No fix needed continuing: {e}")
    return images

def preprocess_multi_page_tiff(image_path):
    images = process_tiff_pages(image_path)
    preprocessed_images = []
    for img in images:
        # Apply preprocessing (example uses preprocessing function)
        processed = remove_noise_and_smooth(img)  # Replace with any preprocessing function
        preprocessed_images.append(processed)
    return preprocessed_images

# Extract text found in the image and write to a text file
def extract_text_from_folder(input, output):
    '''# Iterate over all files in the image folder
    for file_name in os.listdir(input):
        # Check if the file is a TIFF image
        if file_name.endswith(".tif"):
            # Construct the full path to the image file
            image_path = os.path.join(input, file_name)    
            # Extract text from the image
            # Perform OCR using pytesseract
            #? newest image dection and smoothing
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = preprocess_multi_page_tiff(image)
            # img = remove_noise_and_smooth(image)          # %42.2
            
            #* NOTE proprocessing2 the blue needs uncommenting before other test
            text = pytesseract.image_to_string(img) #needed for preprocessing 
                      
            # Calls display funcation
            # show_detected_text(image_path)
            # show_detected_text_from_image(img)
            
            # Construct the full path to the text file
            text_file_name = os.path.splitext(file_name)[0] + ".txt"
            text_file_path = os.path.join(output, text_file_name)
            # Write the extracted text to the text file
            with open(text_file_path, 'w') as text_file:
                text_file.write(text)'''
     # Iterate over all files in the input folder
    for file_name in os.listdir(input):
        # Construct the full path to the file
        image_path = os.path.join(input, file_name)

        # Check if the file is a TIFF image
        if file_name.endswith(".tif"):
            print(f"Processing TIFF file: {file_name}")
            # Preprocess multi-page TIFF
            preprocessed_images = preprocess_multi_page_tiff(image_path)

            # Initialize a variable to store the full text for the file
            full_text = ""

            # Process each page
            for page_index, preprocessed_image in enumerate(preprocessed_images):
                # Perform OCR on the preprocessed image
                text = pytesseract.image_to_string(preprocessed_image)
                full_text += f"\n--- Page {page_index + 1} ---\n{text}"

            # Write the extracted text to a text file
            text_file_name = os.path.splitext(file_name)[0] + ".txt"
            text_file_path = os.path.join(output, text_file_name)
            with open(text_file_path, 'w') as text_file:
                text_file.write(full_text)
        
        else:
            print(f"Skipping unsupported file: {file_name}")
                
if __name__ == "__main__":
    input = ".\\PennTAP history\\unnamed_file"
    output = ".\\PennTAP history\\unnamed_file\\Textfiles"
    extract_text_from_folder(input,output)