import os
import sys
import stat
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import punkt
from nltk import sent_tokenize, FreqDist
from collections import Counter
import shutil as sh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to extract keywords from text
# n is the number of keywords should be taken from passage
def extract_keywords(text, n=10):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    
    #Stemmer removes prefix/suffix from words. Lemmatization looks for the meaning of words and chance it to its simplest form.
    #which would be better for extraction? Lemmatization because it can reduce the noise and variability, making it better for text recognition.
    
    #! removes the prefix/suffix from words making them unrecognized. 
    '''# Create a PorterStemmer object 
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(n)]'''
    
    # Create a WordNetLemmatizer object 
    lemmatizer = WordNetLemmatizer() 
    words = [lemmatizer.lemmatize(word) for word in words]
    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(n)]
    
    
    return keywords

# Function to move files based on keywords
def move_files(input_folder, output_folder, manual_review_folder, image_folder):
    new_filename, txt_file, text = read_text_file_and_rename_image(input_folder)
    if  new_filename != '':  # if the newfile name doesn't exist then more the file into the manual review folder
        new_filename += new_filename + ".tif"
        new_filepath = os.path.normpath(os.path.join(output_folder, new_filename))
        sh.move(os.path.normpath(os.path.join(image_folder, txt_file)), new_filepath)
    else:
        sh.move(os.path.normpath(os.path.join(image_folder, txt_file)), manual_review_folder)
#TODO TypeError in sh.move statements
''' for filename in os.listdir(input_folder):
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
                sh.move(os.path.join(image_folder, filename.replace(".txt",".tif")), manual_review_folder)'''
#*Compared to online summarizer

#Which would be better extraction or abstractive text summarization?
#Abstract give better results for the first test case

#! Replaced with Abstract text summarization
def Esummarize_text(text):
    #create work frequency table
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    
    #breaks passage into sentence
    sentences = sent_tokenize(text)                 
    sentenceValue = dict()                     

    #for every word in a sentence track it's frequency
    #[:10] grabs the first 10 words. this will save memory on longer passages
    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:30] in sentenceValue:
                    sentenceValue[sentence[:30]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:30]] = freqTable[wordValue]

        sentenceValue[sentence[:30]] = sentenceValue[sentence[:30]] // word_count_in_sentence                   

    #find the average frequency of all words in the text to find sumary
    #* might be better to use abstractive text summarization
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    if len(sentenceValue) != 0:
        average = int(sumValues / len(sentenceValue))
    else:
        average = 0
    
    sentence_count = 0
    summary = ''

    #choose the top 3 sentences based on their frequency
    for sentence in sorted(sentenceValue, key=sentenceValue.get, reverse=True):
        if sentence[:30] in sentenceValue and sentenceValue[sentence[:30]] > (1.2 * average) and sentence_count < 3:
            summary += " " + sentence
            sentence_count += 1
    
    return summary

def Asummarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Check if the text contains enough non-stop words for summarization
    non_stop_words_exist = any(word not in stopwords.words('english') for word in word_tokenize(text))

    if not non_stop_words_exist:
        return "Text does not contain enough meaningful content for summarization."
    
    # Create a TF-IDF vectorizer
    #Term Frequency-Inverse document Frequency
    vectorizer = TfidfVectorizer(stop_words='english')

    # Calculate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate the pairwise cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Initialize sentence scores
    sentence_scores = np.zeros(len(sentences))

    # Calculate the score for each sentence by summing cosine similarities
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sentence_scores[i] += cosine_similarities[i][j]

    # Get the indices of the top sentences based on scores
    top_sentence_indices = np.argsort(sentence_scores)[::-1][:3]

    # Create the summary by combining top sentences
    summary = ' '.join([sentences[idx] for idx in sorted(top_sentence_indices)])

    return summary

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
    date_match = re.search(r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})\s+\d{1,2},?\s+\d{4})|(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', text, re.IGNORECASE)
    if date_match:     
        date_str = date_match.group()

        # Replace month names with their number form
        for month_name, month_number in month_map.items():
            date_str = date_str.replace(month_name, month_number)

        # Extract year, month, and day
        year, month, day = re.findall(r'\d+', date_str)
        # Ensure month and day are zero-padded if necessary
        month = month.zfill(2)
        day = day.zfill(2)
        
        summary['date'] = f"{year}/{month}/{day}"
        
    # Extract publisher using regex
    news_match = re.search(r'(?:article|news|newspaper|paper|press|journal)\s+(?:\w+\s+)*', text, re.IGNORECASE)
    if news_match:
        summary['publisher'] = news_match.group()    

    return summary

def read_text_file_and_rename_image(text_file_path):
    # returns new_filename, txt_file at the iteration, and text summary. 
    # Read text from the text file line by line
    # There was an error where it would read the whole text file as one string without \n characters.
    #   This caused the problem of the regex being able to match more than it should have.
    # Iterate over all files in the image folder
    
    # when given a path to a folder, iterate through the contents if it is a text file. 
    for file_name in os.listdir(text_file_path):
        if file_name.endswith(".txt"):  # Check if the file is a text file
            file_path = os.path.join(text_file_path, file_name)
    
            with open(file_path, 'r') as text_file:
                summary = {} 
                for line in text_file:
                    line_summary = extract_summary_from_text(line)
                    summary.update(line_summary)

                #sends summary to the Asummarize_text() and extract_keywords()
                #text = text_file.read()
                
            print("START\n")

            # Construct a new file name based on the summary
            new_file_name = ""
            if 'date' in summary:
                new_file_name += summary['date'] + "_"
            else:
                new_file_name = ''
                print("No date found for file name")
            
            #print(new_file_name)    
            
            if 'publisher' in summary:
                new_file_name += summary['publisher'] + "_"
            else:
                new_file_name = ''
                print("No publisher found for file name")                
            
            #* keyword summary can be done with the function to find the summary and then the nltk to find the keywords of the summary. 
            #*  The threshold should be at most 5 words. 
            
            #TODO Summary needs to be shorted again and then added to the new_file_name
            
            with open(file_path, "r") as summary:
                text = summary.read()
            
            print("Summarized using abtract: \n",Asummarize_text(text))   #give summarize of text
            print("Extracted words from aesum: \n",extract_keywords(Asummarize_text(text))) #gathers 15 keywords
            print("END\n",new_file_name,"\n\n")
            
            return new_file_name, text_file, text # Return the new file, text_file location, and text gathered. 
    return "", "", ""  # Return empty strings if no text files were found

if __name__ == "__main__":
    input_folder = ".\\unnamed_file\\Textfiles"
    image_folder = ".\\unnamed_file"
    output_folder = ".\\complete_images"
    manual_review_folder = ".\\manual_review_images"
    # Move files based on keywords
    move_files(input_folder, output_folder, manual_review_folder, image_folder)
    #read_text_file_and_rename_image(input_folder)
    
    #TODO Set up file movement by returning file name to move_file()