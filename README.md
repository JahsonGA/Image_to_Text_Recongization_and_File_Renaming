# Text Processing and Image File Management

This project includes Python scripts for text processing tasks such as keyword extraction, text summarization, and file operations for managing images based on extracted keywords.

# Dev log
### 4/12/24.
Created a virtual environment to host the data for the image processing algorithm.  Install tesseract, opencv, numpy, PIL, os, re, Pandas, and nltk. I started work on developing an algorithm to detect text on a file and renaming the file to the text found. However there is a lot of text can be found. So to shorten the text found by summarizing the key and searching to keywords or ideas with the file with an abstract text summarization, regex expressions, and word search functions. First the word search looks for dates in the format of month, day, year; day month year; or month year format. If the month and year is found but no date the date is filled in with zeros. Next the word search looks for publishers that follow the format of "type: (key news word)". type is defined as articles, newspapers, press, journals, etc. Key news words are define as any sentence that has "times, post, day, tribune etc." following the type. If either of these two thing can be found in the file then mark it to be moved to manual review. Next the abstract text summarization is called to taken the given found text and summarize the information by look at the amount of non-stop words in English and a Term Frequency-Inverse Document Frequency (FT-IDF) matrix. On the matrix a pairwise cosine function can be used to find similarity between the words found and the words that exist in the training data. The key words found are limited to 15 and stored in the new life name. From the words that score the highest base on frequency and similarity the key ideas for the summary can be found. Finally after all this calculations have been made if the enough words have found move the file to its new location. If file name has enough to quantify a success, then more it to the completed folder. Otherwise the file will be moved to the manual review folder. 
 
After doing a test on the image processing, I discovered that a large amount of files were being marked as manual review folder. I calculated a 33% success rate to that current image processing. To improve the image processing. I created many function that apply different effects to a file. These effects can include rotating image, image scaling,  noise removal (noise is define as the grainy or pixelization of an image), gray scaling, thresholding or binarization, and in the additional "flags" to check whether an operation needed. For each image file there is currently two functions to apply image preprocessing. In both the process is similar. the image is read into an object and then the contrast is increased by converting image to gray scale and then equalize the the pixels based on their neighbors. This creates a  sharp edge in case it was lost when scanning. Next the image is converted to a binary format. This means that for every pixel is black (off) or white (on). Otsu's Thresholding is apply on the binary image. This means that a Gaussian Blur is need to help create a generate sharp outline of all the binary files. Depending on the process needed the image could either create a enhance the text region and make them larger and remove any non-consistent pixels in the letters or a create a mask over the text regions to help find text. The mask will focus the image to where there is text and dilated the text and remove non-consistent pixels. Then join the image back together. Image are then scanned for any recognized text and it is written to a text file to be analyzed later. There is also a helper function to show what text was found by draw a box around any letters found. 
 
I am current working on improve the image preprocessing to ensure that the most about of file can be found at one time. Currently the highest the algorithm has scored is 61%. I want to try noise reduction to improve text detection. As of now it is only returns a success score of 58% with a sample size of 92.  

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install NLTK.
### On Windows
```
pip install pytesseract
```
### On Linux
```
sudo apt-get install tesseract-ocr
```

### Install opencv
```
pip install opencv-python
```

### Install numpy
```
pip install numpy
```

### Install PIL (Python Imaging Library)
```
pip install Pillow
```

### Install pandas
```
pip install pandas
```

### Install nltk
```
pip install nltk
```

# Usage
Use the `ExtractText.py` to gather the text and `FindKeyData.py` to filter the files based on the text of the extracted data.
## ExtractText

### Deskew
`Skew Correction`
While scanning or taking a picture of any document, 
it is possible that the scanned or captured image might be slightly skewed sometimes. 
For the better performance of the OCR, it is good to determine the skewness in image and correct it.

### Set Image DPI
`set_image_dpi`
To achieve a better performance of OCR, 
the image should have more than 300 PPI (pixel per inch). 
So, if the image size is less than 300 PPI, we need to increase it. 
We can use the Pillow library for this.

### Noise Removal
`remove_noise`
This step removes the small dots/patches 
which have high intensity compared to the rest of the image for smoothening of 
the image. OpenCV's fast Nl Means Denoising Coloured function can do that easily.

### Gray Scale image
`get_grayscale`
This process converts an image from other color spaces to shades of Gray. 
The colour varies between complete black and complete white. 
OpenCV's cvtColor() function perform this task very easily.

### Thresholding or Binarization
`thresholding`
This step converts any colored image into a binary image that contains only two colors black and white. 
It is done by fixing a threshold (normally half of the pixel range 0-255, i.e., 127). 
The pixel value having greater than the threshold is converted into a white pixel else into a black pixel. 
To determine the threshold value according to the image Otsu's Binarization and Adaptive Binarization can be a better choice. 
In OpenCV, this can be done as given.

### Counts White Pixels
`get_white_pixel_count`

### Noise Reduces
`needs_noise_reduction`
Helper function for determining which image processing algorithm to use

### Show text found
`show_detected_text`
Shows the edited image with boxes will the letter opencv sees. Given a image path or image object it will create.

### Preprocessing
There are two function used for image preprocessing. The only difference is how the preprocessing is preformed. Both need an image path

`preprocess_for_ocr`
Increase Contrast, Calculate white pixel count for binary and inverted binary images, Choose the 'better' image based on white pixel count. Once the pixel count has been calculated and the image is in binary/invert binary form, then the Gaussian blur is applied. The threshold is then calculated for image to define edge in the image. Next remove small objects found within the image and enhance the image by dialing the text to make it larger and more recognizable. Finally pixel are removed to sharpen the image and the resolution increases. 
`preprocessing`
Similar to the function above, after the image contrast is increased and the image is converted a binary image. But afterwards the Otsu's thresholding and Gaussian filtering is done on the image. This information is used to calculate a mask over all the places with text and the text is dilated. The image is bitwise and operation is done to combine the mask and binary image to find the text. 

### Extract text
Extract text found in the image and write to a text file when given a path to a folder containing images and a path to a folder for the text files to be sent.
`extract_text_from_folder` 
Extract text found in the image and write to a text file. Iterate over all files in the image folder:
Check if the file is a TIFF image. If it has been:
    Construct the full path to the image file
    Extract text from the image
    Perform OCR using pytesseract
    Calls display funcation
    Construct the full path to the text file
    Write the extracted text to the text file

## FindKeyData

### Keyword Extraction
Use the `extract_keywords` function to extract keywords from a given text.

The function to extract keywords from given text and find n number of key words from the passage. Function return a list of keywords.
Note:
Stemmer removes prefix/suffix from words. Lemmatization looks for the meaning of words and chance it to its simplest form.
which would be better for extraction? Lemmatization because it can reduce the noise and variability, making it better for text recognition.

### Text Summarization

Use the `Asummarize_text` function to summarize text. Summarization uses abstractive text summarization. 

The function take a list of text and return a list of that same text but it is summarized by using an abstractive text summarization. Abstractive summarization is used to summarize text because it uses a natural language techniques to interpret and understand the important aspects of a text and generate a more “human” friendly summary. While an Extractive summarization involves identifying important sections from text and generating them verbatim which produces a subset of sentences from the original text. [read more here](https://blog.paperspace.com/extractive-and-abstractive-summarization-techniques/). 


### File Operations

Use the `move_files` function to rename and move files based on keywords found.

This function need a path to text files folder gathered from `ExtractText`, path to success folder, path to where the manual review images should be sent, and path to images folder. This function is going to read the text from an image with the `read_text_file_and_rename_image` function. If `read_text_file_and_rename_image` function marked it as correct then it will be moved to the success folder given, otherwise it will be moved to a manual review folder. At the end the text file used for the renaming will be removed.

### Keyword Extraction: 

The `extract_keywords_from_text` function given a summarized text file, the function used a regular expression to find words used in dates or publication. Then a list holds the information gathered from the regular expression from `read_text_file_and_rename_image` function. 

the `read_text_file_and_rename_image` function given a text file path, returns new_filename, txt_file at the iteration, and text summary. Read text from the text file line by line and run `Asummarize_text` to build the summary. Construct a new file name based on the summary: If there is a date in the summary add to the file name
If there is a publisher in the summary add it file name
If there is already a date in the file name then skip it


# Resourced Used
https://nanonets.com/blog/ocr-with-tesseract/
https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
https://www.educative.io/answers/tesseract-text-detection-in-computer-vision
https://stackoverflow.com/questions/54246492/pytesseract-difference-between-image-to-string-and-image-to-boxes
https://pypi.org/project/pytesseract/
https://stackoverflow.com/questions/56698912/what-does-the-key-values-of-the-dictionary-output-of-the-following-code-in-tesse
https://stackoverflow.com/questions/76834972/how-can-i-run-pytesseract-python-library-in-ubuntu-22-04
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
https://www.analyticsvidhya.com/blog/2023/03/getting-started-with-image-processing-using-opencv/
https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c
https://www.geeksforgeeks.org/image-enhancement-techniques-using-opencv-python/#google_vignette
https://neptune.ai/blog/image-processing-python
https://opencv.org/
https://docs.opencv.org/4.x/d7/da8/tutorial_table_of_content_imgproc.html
https://www.geeksforgeeks.org/reading-image-opencv-using-python/
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
https://stackoverflow.com/questions/22265051/how-to-remove-salt-and-pepper-noise-from-images-using-python
https://learnopencv.com/otsu-thresholding-with-opencv/
https://www.projectpro.io/recipes/remove-noise-from-images-opencv
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
