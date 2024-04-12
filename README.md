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

## Usage

### Keyword Extraction
Use the `extract_keywords` function to extract keywords from a given text.

```python
from keyword_extraction import extract_keywords

text = "Sample text for keyword extraction."
keywords = extract_keywords(text, n=5)
print(keywords)
```

### Text Summarization

Use the `Asummarize_text` function to summarize text extractively.

```python
from text_summarization import Asummarize_text

text = "Sample text for summarization."
summary = Asummarize_text(text)
print(summary)
```

### File Operations

Use the `move_files` function to move files based on keywords.

```python
from file_operations import move_files

input_folder = "input_folder_path"
output_folder = "output_folder_path"
manual_review_folder = "manual_review_folder_path"
image_folder = "image_folder_path"

move_files(input_folder, output_folder, manual_review_folder, image_folder)
```

## Text Processing:

### Keyword Extraction: 
The extract_keywords function extracts keywords from a given text using NLTK's tokenization, stop word removal, and lemmatization.
### Text Summarization: 
The Asummarize_text function summarizes text extractively using TF-IDF vectorization and cosine similarity.
### Text Summarization (Alternative): 
The Esummarize_text function provides an alternative summarization method using word frequency and sentence scoring.
## File Operations:

### Move Files Based on Keywords: 
The move_files function moves image files to specified output folders based on keywords extracted from associated text files.
## Other Features:

### File Name Renaming: 
Text files are renamed based on extracted keywords for better organization.
### Date and Publisher Extraction: 
The project can extract dates and publishers from text for file naming purposes.

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
