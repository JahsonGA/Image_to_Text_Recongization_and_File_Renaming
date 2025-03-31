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
    if(new_fileName.isdigit()):
        if (new_fileName[:-2] == "00"):
            # missing month
            if (new_fileName[:-4] == "0000"):
                #only year found. Store the first 4 characters 
                new_fileName = new_fileName[:4] + "_" + new_fileName[4:6] + "_" + new_fileName[6:8]
            
            #date month and year found. store first 8 characters
            else:
                new_fileName = new_fileName[:2] + "_" + new_fileName[2:4] + "_" + new_fileName[4:8]
        
        #month and year found. store first 6 characters
        else:
            new_fileName = new_fileName[:2] + "_" + new_fileName[2:6] + "_" + new_fileName[6:]

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
    #! older keyword filer
    # words = word_tokenize(text.lower())
    # stop_words = set(stopwords.words('english'))
    # words = [word for word in words if word.isalnum() and word not in stop_words]
    
    #Stemmer removes prefix/suffix from words. Lemmatization looks for the meaning of words and chance it to its simplest form.
    #which would be better for extraction? Lemmatization because it can reduce the noise and variability, making it better for text recognition.
    
    #! removes the prefix/suffix from words making them unrecognized. 
    '''# Create a PorterStemmer object 
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(n)]'''
    
    # Create a WordNetLemmatizer object 
    # lemmatizer = WordNetLemmatizer() 
    # words = [lemmatizer.lemmatize(word) for word in words]
    '''word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(n)]'''
    # bi_grams = ngrams(words, 2)
    # word_freq = Counter(words + [' '.join(bg) for bg in bi_grams])
    # keywords = [word for word, _ in word_freq.most_common(n)]
    # 
    # return keywords

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
            
    '''
    count = 0
    for image_name in os.listdir(image_folder):
        # Sends the text files to be read and stores the new filename, txt_file location, and text summary
        #file_package = read_text_file_and_rename_image(input_folder)
        #package_values = file_package.values()
        #print("Iteration: ", count, "\nLocation: ", list(package_values[0][1]), "\nNew filename: ", list(package_values[0][0]), "\nText: ", list(package_values[0][2]))
    
        new_filename, txt_file, text = read_text_file_and_rename_image(input_folder)
    
        if image_name.endswith(".tif"):
            #print("Iteration: ", count+1)
            if  new_filename != '' and new_filename[0] != '_':  # if the newfile name doesn't exist then more the file into the manual review folder
                new_filename = new_filename + ".tif"
                #new_filename = word_segmenter(new_filename, word_list)
                new_filepath = os.path.normpath(os.path.join(output_folder, new_filename))
                # sh.move(os.path.normpath(os.path.join(image_folder,image_name)), os.path.normpath(new_filepath))
                #print("Scr: ", os.path.normpath(os.path.join(image_folder,image_name)), "\tDst: ", os.path.normpath(new_filepath))
                completedCount += 1
            else:
                # sh.move(os.path.normpath(os.path.join(image_folder,image_name)), os.path.normpath(os.path.join(manual_review_folder,image_name)))
                #print("Manual\nScr: ", os.path.normpath(os.path.join(image_folder,image_name)), "\tDst: ", os.path.normpath(os.path.join(manual_review_folder,image_name)))
                manualCount += 1
            
            #? this line will have to be put in the FileOrg file
            #os.remove(os.path.normpath(os.path.join(input_folder, txt_file)))
        count += 1
        '''
        
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

'''
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Check if the sentences contain meaningful content
    if not sentences or all(contains_only_stop_words(sentence) for sentence in sentences):
        return "Text does not contain enough meaningful content for summarization."

    # Check if the text contains enough non-stop words for summarization
    #non_stop_words_exist = any(word not in stopwords.words('english') for word in word_tokenize(text))

    #if not non_stop_words_exist:
    #    return "Text does not contain enough meaningful content for summarization."
    
    # Create a TF-IDF vectorizer
    #Term Frequency-Inverse document Frequency
    vectorizer = TfidfVectorizer(stop_words='english')

    try:
        # Calculate the TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return "Failed to create TF-IDF matrix; input text might be empty or invalid."
    
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
'''

#* Naming system should follow date, publisher. the key summary will be done the file renaming
def extract_summary_from_text(text):
    '''
    summary = {}
    date_match = re.search(r'(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)|(\b\w+ \d{1,2}, \d{4}\b)', text)
    publisher_match = re.search(r'(article|news|journal|newspaper|press) ([A-Za-z ]+)', text, re.IGNORECASE)
    
    if date_match:
        summary['date'] = date_match.group()
    if publisher_match:
        summary['publisher'] = publisher_match.group(2).strip()

    return summary
    
    '''
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
            # print(date_str)
        # print("Final str: ",date_str)

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
'''
    new_file_name = ""
    
    # Add date to the filename if it exists
    if 'date' in summary:
        new_file_name += summary['date'] + "_"
    
    # Add publisher to the filename if it exists
    if 'publisher' in summary:
        new_file_name += summary['publisher'] + "_"

    # Add keywords extracted from the final summary
    keywords = extract_keywords(final_summary)
    for keyword in keywords:
        new_file_name += keyword + "-"
        
    # replace spaces with - and remove any special symbols
    new_file_name = re.sub(r'[<>:"/\\|?*]', '_', new_file_name)
    new_file_name = new_file_name.replace('\n', '')#.replace('-', '')
    new_file_name = new_file_name[:100] #trim to 100 characters

    return new_file_name.rstrip('-')  # Remove any trailing dash
'''
    
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
    '''
    for file_name in os.listdir(text_file_path):
        if file_name.endswith(".txt") and file_name not in completed_files:
            file_path = os.path.join(text_file_path, file_name)
            with open(file_path, 'r') as text_file:
                text = text_file.read()
                
            if contains_only_stop_words(text):
                print("File contains only stop words. Skipping summarization.")
                return "Stop words only", file_name, text

            summary = extract_summary_from_text(text)
            initial_summary = Asummarize_text(text)
            contextual_summary = SummaryEnchanced(text)

            # Combine summaries or use one as needed
            final_summary = contextual_summary  # or combine both
            
            # Use one of the summaries or combine them
            if contextual_summary:
                final_summary = contextual_summary
            else:
                final_summary = initial_summary  # Fallback to the initial summary

            new_file_name = generate_filename(summary,final_summary)

            # Process the file based on the new filename
            # Move or save the file with new_file_name
            completed_files.append(file_name)

            return new_file_name, file_name, text

    return "", "", ""
    '''

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
    #print("Success: ", completedCount, "Fail: ", manualCount)
    #print("Success Rate: ", (completedCount / (completedCount + manualCount)) * 100)