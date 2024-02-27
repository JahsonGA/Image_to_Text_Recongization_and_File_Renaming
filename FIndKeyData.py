import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import punkt
from collections import Counter

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
def move_files(input_folder, output_folder, manual_review_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), 'r') as txt_file:
                content = txt_file.read()
                keywords = extract_keywords(content)
                if len(keywords) >= 3:  # Set the threshold for minimum keywords required
                    new_filename = "_".join(keywords) + ".txt"
                    new_filepath = os.path.join(output_folder, new_filename)
                    os.move(os.path.join(input_folder, filename), new_filepath)
                else:
                    os.move(os.path.join(input_folder, filename), manual_review_folder)
                    
# TODO
#os move funciton isn't working properly. 
#renames the text files and not the image file
#renaming the file attaches all keywords to the file
#   algorithm need to learn what keywords are more valuable then others. 


if __name__ == "__main__":
    input_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\unnamed_file\\Textfiles"
    output_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\complete_images"
    manual_review_folder = "C:\\Users\\Owner\\OneDrive\\Desktop\\Coding Projects\\PennTap projects\\PennTAP history\\manual_review_images"
    # Move files based on keywords
    move_files(input_folder, output_folder, manual_review_folder)