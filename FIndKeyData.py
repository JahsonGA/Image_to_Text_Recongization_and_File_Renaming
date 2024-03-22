import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
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
    #failed attempt to get move function to work correctly
    #cwd = os.getcwd()  # Get the current working directory (cwd)
    #files = os.listdir(cwd)  # Get all the files in that directory
    #print("Files in %r: %s" % (cwd, files))
    #input_folder = cwd + input_folder
    #output_folder = cwd + output_folder
    #manual_review_folder = cwd + manual_review_folder
    
    count = 0
    for image_name in os.listdir(image_folder):
        # Sends the text files to be read and stores the new filename, txt_file location, and text summary
        file_package = read_text_file_and_rename_image(input_folder)
        #TODO correct the file unpacking
        package_values = file_package.values()
        print("Iteration: ", count, "\nLocation: ", list(package_values[0][1]), "\nNew filename: ", list(package_values[0][0]), "\nText: ", list(package_values[0][2]))
    
        if image_name.endswith(".tif"):
            
            if  new_filename != '':  # if the newfile name doesn't exist then more the file into the manual review folder
                new_filename = new_filename + ".tif"
                new_filepath = os.path.normpath(os.path.join(output_folder, new_filename))
                #sh.move(os.path.normpath(os.path.join(image_folder,image_name)), os.path.normpath(new_filepath))
                print("Dst: ", os.path.normpath(new_filepath), "Scr: ", os.path.normpath(os.path.join(image_folder,image_name)))
            else:
                #sh.move(os.path.normpath(os.path.join(image_folder,image_name)), os.path.normpath(os.path.join(manual_review_folder,image_name)))
                print("Manual")
        count += 1
        
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
    date_match = re.search(r'(?:(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})\s+\d{1,2},?\s+\d{4})|(?:\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})', str(text), re.IGNORECASE)
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
        
        summary['date'] = f"{year}-{month}-{day}"
        
    # Extract publisher using regex
    news_match = re.search(r'(?:article|news|newspaper|paper|press|journal)\s+(?:\w+\s+)*(?:times|post|today|day|tribune|globe|news|newspaper|paper|press|journal)', str(text), re.IGNORECASE)
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
    completedArry = {}
    for file_name in os.listdir(text_file_path):
        if file_name.endswith(".txt"):  # Check if the file is a text file
            file_path = os.path.join(text_file_path, file_name) # Creates file path to txt
            new_file_name = ""
            
            with open(file_path, 'r') as text_file:
                summary = {} 
                strTextFile = text_file.read()
                for line in strTextFile:
                    summary = extract_summary_from_text(line)
                    
                    # Construct a new file name based on the summary
                    if 'date' in summary:
                        new_file_name += summary['date'] + "_"
                        
                        if 'publisher' in summary:
                            new_file_name += summary['publisher'] + "_"
                            break
                        else:
                            #print("No publisher found for file name in line:\n\t", line)
                            continue
                        
                    else:
                        #print("No date found for file name in line:\n\t", line)
                        continue
                    
                text_file.close()
            
            text_file.close()
                
            # print("START\n")              
            
            #* keyword summary can be done with the function to find the summary and then the nltk to find the keywords of the summary. 
            #*  The threshold should be at most 5 words. 
            
            with open(file_path, "r") as summary:
                text = summary.read()
                summary.close()
            summary.close()
            
            aText = Asummarize_text(text)
            extr_aText = extract_keywords(Asummarize_text(text))
            extr_eText_aText = extract_keywords(Esummarize_text(Asummarize_text(text)))
            
            # print("Summarized using abtract: \n",aText)   #give summarize of text
            # print("Extracted words from asum: \n",extr_aText) #gathers 15 keywords
            # print("Extracted words from esum of asum: \n",extr_eText_aText) #gathers 5 keywords
            
            if (len(extr_aText) >= len(extr_eText_aText)):
                for i in extr_aText:
                    new_file_name += i + "-"
            
            elif (len(extr_aText) < len(extr_eText_aText)):
                for i in extr_eText_aText:
                    new_file_name += i + "-"
                    
            if new_file_name == "":
                new_file_name = ''      #if the new file name is empty then set it to empty
            else:
                new_file_name = new_file_name.rstrip(new_file_name[-1])
                #new_file_name += "_"    #otherwise end the file name with a "_"
            
            # print("\nNew File name: ",new_file_name,"\nEND")
            
            completedArry[file_name] = [new_file_name, file_name, text]
            
        return completedArry # Return the new file, text_file location, and text gathered. 

    return "", "", ""  # Return empty strings if no text files were found

if __name__ == "__main__":
    input_folder = ".\\unnamed_file\\Textfiles"
    image_folder = ".\\unnamed_file"
    output_folder = ".\\complete_images"
    manual_review_folder = ".\\manual_review_images"
    # Move files based on keywords
    move_files(input_folder, output_folder, manual_review_folder, image_folder)
    #read_text_file_and_rename_image(input_folder)
    
    #TODO correct move_file() to work with new file name