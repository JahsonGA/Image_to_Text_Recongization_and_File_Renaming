import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import ngrams
from nltk.corpus import words
from collections import Counter
import shutil as sh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

#globals
completed_files = []
completedCount = 0
manualCount = 0
nlp = spacy.load('en_core_web_sm')

# Get the list of valid English words
word_list = words.words()

# Helper function to segment concatenated words
def word_segmenter(fileName, word_list):
    n = len(fileName)                   #length of fileName
    dp = [None] * (n+1)                 #defines the array size
    dp[0] = []                          #Base case: empty string can be segmented into an empty list of words
    new_fileName = fileName[:8]         #gets the first 8 characters which should hold dates
    fileName = fileName[8:]             #removes the first 8 characters
    #print(word_list)
    
    # missing date
    if new_fileName.isdigit():
        if new_fileName[:4] == "0000":
            # Missing year (unlikely, but handle it)
            new_fileName = "unknown_date"
        elif new_fileName[4:6] == "00":
            # Missing month, only year is present
            new_fileName = f"{new_fileName[:4]}_00_00"
        elif new_fileName[6:8] == "00":
            # Missing day, month and year are present
            new_fileName = f"{new_fileName[:4]}_{new_fileName[4:6]}_00"
        else:
            # Date month and year are present
            new_fileName = f"{new_fileName[:4]}_{new_fileName[4:6]}_{new_fileName[6:8]}"

    # Iterate over the string and segment it based on word_list
    for i in range(1, n + 1):
        for j in range(i):
            word = fileName[j:i]
            if word in word_list and dp[j] is not None:
                dp[i] = dp[j] + [word]
                break
    
    # Return the segmented words if possible, otherwise return an empty string
    if dp[-1] is not None: 
        # Ensure correct formatting with underscores between words
        segmented_words = '_'.join(dp[-1])
        new_fileName += '_' + segmented_words
    else:
        new_fileName += fileName
        
    new_fileName = new_fileName.replace(" ","_")
    
    return new_fileName

# Function to extract keywords from text
# n is the number of keywords should be taken from passage
def extract_keywords(text, n=10):
    
    doc = nlp(text)
    keywords = [ent.text for ent in doc.ents if ent.label_ in ('DATE', 'ORG', 'PERSON', 'EVENT')]
    
    if len(keywords) < n:
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.isalnum() and word not in stop_words]
        keywords += words[:n - len(keywords)]

    return keywords[:n]

# Function to move files based on keywords
def move_files(input_folder, output_folder, manual_review_folder, image_folder):    
    global completedCount, manualCount
    
    for image_name in os.listdir(image_folder):
        if image_name.endswith(".tif"):
            new_filename, txt_file, text = read_text_file_and_rename_image(input_folder)
            
            if new_filename and new_filename != 'EmptyText':
                new_filename = new_filename + ".tif"
                new_filepath = os.path.normpath(os.path.join(output_folder, new_filename))
                sh.move(os.path.normpath(os.path.join(image_folder, image_name)), new_filepath)
                completedCount += 1
            else:
                sh.move(os.path.normpath(os.path.join(image_folder, image_name)), os.path.normpath(os.path.join(manual_review_folder, image_name)))
                manualCount += 1

def contains_only_stop_words(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())

    # Check if all words are stop words
    all_stop = all(word in stopwords.words('english') for word in words)

    return all_stop

def Asummarize_text(text):
    
    sentences = [sent.text for sent in nlp(text).sents]
    
    if not sentences:
        return "No meaningful content"

    vectorizer = TfidfVectorizer(stop_words='english')

    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return "Empty or invalid text"

    sentence_scores = np.sum(cosine_similarity(tfidf_matrix, tfidf_matrix), axis=1)
    top_indices = np.argsort(sentence_scores)[::-1][:3]

    return ' '.join([sentences[i] for i in sorted(top_indices)])

#* Naming system should follow date, publisher. the key summary will be done the file renaming
def extract_summary_from_text(text):
   
    summary = {}
    month_map = {
        'January': '01',
        'February': '02',
        'March': '03',
        'April': '04',
        'May': '05',
        'June': '06',
        'July': '07',
        'August': '08',
        'September': '09',
        'October': '10',
        'November': '11',
        'December': '12'
    }

    # Extract dates using regex
                                            #vv Month,Day year vv                                                                                                                vv day month year vv                                                                                 vv month year vv day month year vv                                                          
    date_match = re.search(r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})\s+\d{1,2},?\s+\d{4})|(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})|(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}', str(text), re.IGNORECASE)
    if date_match:     
        date_str = date_match.group()
        #print(date_str)

        # Replace month names with their number form
        for month_name, month_number in month_map.items():
            date_str = date_str.replace(month_name, month_number)

        # Extract year, month, and day
        date_values = re.findall(r'\d+', date_str)
        year = date_values[0]
        month = date_values[1] if len(date_values) > 1 else '00'  # Default to '01' if month is missing
        day = date_values[2] if len(date_values) > 2 else '00'    # Default to '01' if day is missing
        
        # Ensure month and day are zero-padded if necessary
        month = month.zfill(2)
        day = day.zfill(2)
        
        summary['date'] = f"{year}-{month}-{day}"

        
    # Extract publisher using regex
    news_match = re.search(r'(?:\b(?:article|news|newspaper|paper|press|journal)\b\s+(?:[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*))|(?:\b(?:THE|A|AN)?\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*(?:\s+NEWSPAPER)?\b)', str(text), re.IGNORECASE)
    if news_match:
        summary['publisher'] = news_match.group()

    return summary
    

def SummaryEnchanced(text):
    #combines the abstractive summary with the keywords found. Should boost overall success rate
    extract_summary = Asummarize_text(text)     # Abstractive
    keywords = extract_keywords(text, n=5)      # Extractive keywords

    # Combine extractive summary with keywords
    combined_summary = extract_summary + " " + " ".join(keywords)
    
    return combined_summary

def generate_filename(summary, text):
    
    date = summary.get('date', 'UnknownDate').replace('/', '-').replace(',', '')
    publisher = summary.get('publisher', 'UnknownPublisher').replace(' ', '_')
    
    keywords = extract_keywords(text)
    keywords_part = '_'.join(keywords)

    filename = f"{date}_{publisher}_{keywords_part}"
    filename = re.sub(r'[^\w\-]', '', filename)

    return filename[:100]  # Limit filename length
    
def read_text_file_and_rename_image(text_file_path):
    
    for file_name in os.listdir(text_file_path):
        if file_name.endswith(".txt") and file_name not in completed_files:
            file_path = os.path.join(text_file_path, file_name)
            with open(file_path, 'r') as text_file:
                text = text_file.read()
            
            if len(text.strip()) == 0:
                return "EmptyText", file_name, text

            summary = extract_summary_from_text(text)
            final_summary = Asummarize_text(text)
            new_file_name = generate_filename(summary, final_summary)

            completed_files.append(file_name)
            return new_file_name, file_name, text

    return "", "", ""

if __name__ == "__main__":
    input_folder = ".\\PennTAP history\\unnamed_file\\Textfiles"
    image_folder = ".\\PennTAP history\\unnamed_file"
    output_folder = ".\\PennTAP history\\complete_images"
    manual_review_folder = ".\\PennTAP history\\manual_review_images"
    # Move files based on keywords
    move_files(input_folder, output_folder, manual_review_folder, image_folder)
    total_files = completedCount + manualCount
    if total_files > 0:
        success_rate = (completedCount / total_files) * 100
        print("Success Rate: ", success_rate)
    else:
        print("No files were processed.")