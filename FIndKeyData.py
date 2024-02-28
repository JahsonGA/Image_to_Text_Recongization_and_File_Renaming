import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import punkt
from collections import Counter
import shutil as sh

# Function to extract keywords from text
def extract_keywords(text, n=5):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(n)]
    return keywords

# Function to move files based on keywords
def move_files(input_folder, output_folder, manual_review_folder, image_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), 'r') as txt_file:
                content = txt_file.read()
                keywords = extract_keywords(content)
                if len(keywords) >= 3:  # Set the threshold for minimum keywords required
                    new_filename = "_".join(keywords) + ".tif"
                    new_filepath = os.path.join(output_folder, new_filename)
                    os.rename(os.path.join(image_folder, filename.replace(".txt",".tif")), os.path.join(output_folder, new_filename))
                    sh.move(os.path.join(image_folder, filename), new_filepath)
                else:
                    sh.move(os.path.join(image_folder, filename.replace(".txt",".tif")), manual_review_folder)
                    
# TODO
#os move funciton isn't working properly. 
#renames the text files and not the image file
#renaming the file attaches all keywords to the file
#   algorithm need to learn what keywords are more valuable then others. 

#TODO
#Compared to online summizer


#TODO try the following of 2/29/24
'''def extract_summary_from_text(text):
    summary = {}

    # Extract dates using regex
    date_match = re.search(r'\b(\d{1,2}\/\d{1,2}\/\d{2,4})\b', text)
    if date_match:
        summary['date'] = date_match.group(1)

    # Extract publisher using regex
    publisher_match = re.search(r'Publisher:\s*(.+)', text)
    if publisher_match:
        summary['publisher'] = publisher_match.group(1)

    # Extract subjects or topics using regex
    subjects_match = re.search(r'Subjects?:\s*(.+)', text)
    if subjects_match:
        summary['subjects'] = subjects_match.group(1).split(',')

    # Extract locations using regex
    locations_match = re.search(r'Locations?:\s*(.+)', text)
    if locations_match:
        summary['locations'] = locations_match.group(1).split(',')

    return summary

def read_text_file_and_rename_image(text_file_path, image_file_path):
    # Read text from the text file
    with open(text_file_path, 'r') as text_file:
        text = text_file.read()
        summary = extract_summary_from_text(text)

    # Construct a new file name based on the summary
    new_file_name = ""
    if 'date' in summary:
        new_file_name += summary['date'] + "_"
    if 'publisher' in summary:
        new_file_name += summary['publisher'] + "_"
    if 'subjects' in summary:
        new_file_name += "_".join(summary['subjects']) + "_"
    if 'locations' in summary:
        new_file_name += "_".join(summary['locations'])

    # Rename the image file
    new_image_file_path = os.path.join(os.path.dirname(image_file_path), new_file_name + ".tif")
    os.rename(image_file_path, new_image_file_path)
'''


if __name__ == "__main__":
    input_folder = ".\\unnamed_file\\Textfiles"
    image_folder = ".\\unnamed_file\\"
    output_folder = ".\\complete_images"
    manual_review_folder = ".\\manual_review_images"
    # Move files based on keywords
    move_files(input_folder, output_folder, manual_review_folder, image_folder)