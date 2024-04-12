# Text Processing and Image File Management

This project includes Python scripts for text processing tasks such as keyword extraction, text summarization, and file operations for managing images based on extracted keywords.

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

```bash
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

### Keyword Extraction: The extract_keywords function extracts keywords from a given text using NLTK's tokenization, stop word removal, and lemmatization.
### Text Summarization: The Asummarize_text function summarizes text extractively using TF-IDF vectorization and cosine similarity.
### Text Summarization (Alternative): The Esummarize_text function provides an alternative summarization method using word frequency and sentence scoring.
## File Operations:

### Move Files Based on Keywords: The move_files function moves image files to specified output folders based on keywords extracted from associated text files.
## Other Features:

### File Name Renaming: Text files are renamed based on extracted keywords for better organization.
### Date and Publisher Extraction: The project can extract dates and publishers from text for file naming purposes.

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