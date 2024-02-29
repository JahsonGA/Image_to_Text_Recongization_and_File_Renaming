import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import punkt
from collections import Counter
import shutil as sh
import re

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
                    
#TODO
#os move function isn't working properly. 
#renames the text files and not the image file
#renaming the file attaches all keywords to the file
#   algorithm need to learn what keywords are more valuable then others. 

#*Compared to online summarizer


def extract_summary_from_text(text):
    summary = {}

    # Extract dates using regex
    date_match = re.search(r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})\s+\d{1,2},?\s+\d{4})|(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', text, re.IGNORECASE)
    if date_match:
        summary['date'] = date_match.group()

    # Extract publisher using regex
    news_match = re.search(r'(?:news|newspaper|paper|press|journal)\s+(?:\w+\s+)*', text, re.IGNORECASE)
    if news_match:
        summary['publisher'] = news_match.group()
        
    #TODO Naming system should follow publisher, date, and keywork summary
    #* keyword summary can be done with the function to find the summary and then the nltk to find the keywords of the summary. 
    #*  The threshold should be at most 5 words. 

    '''# Extract subjects or topics using regex
    subjects_match = re.search(r'Subjects?:\s*(.+)', text)
    if subjects_match:
        summary['subjects'] = subjects_match.group(1).split(',')

    # Extract locations using regex
    locations_match = re.search(r'[A-Z][a-zA-Z]+', text)
    if locations_match:
        summary['locations'] = locations_match.group(1).split(',')'''

    return summary

def read_text_file_and_rename_image(text_file_path):
    # Read text from the text file line by line
    #! There was an error where it would read the whole text file as one string without \n characters.
    #!   This caused the problem of the regex being able to match more than it should have.
    with open(text_file_path, 'r') as text_file:
        summary = {}
        for line in text_file:
            line_summary = extract_summary_from_text(line)
            summary.update(line_summary)

    # Construct a new file name based on the summary
    new_file_name = ""
    if 'date' in summary:
        new_file_name += summary['date'] + "_"
    
    #print(new_file_name)    
    
    if 'publisher' in summary:
        new_file_name += summary['publisher'] + "_"
        
    #print(new_file_name)
    
    '''if 'subjects' in summary:
        new_file_name += "_".join(summary['subjects']) + "_"
    if 'locations' in summary:
        new_file_name += "_".join(summary['locations'])'''
        
    print(new_file_name)

if __name__ == "__main__":
    input_folder = ".\\unnamed_file\\Textfiles\\20231207095006269.txt"
    image_folder = ".\\unnamed_file"
    output_folder = ".\\complete_images"
    manual_review_folder = ".\\manual_review_images"
    # Move files based on keywords
    # move_files(input_folder, output_folder, manual_review_folder, image_folder)
    read_text_file_and_rename_image(input_folder)