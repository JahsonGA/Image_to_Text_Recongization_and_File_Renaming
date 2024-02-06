import os
import re

def extract_metadata_from_text(text):
    metadata = {}

    # Extract title (assuming it is the first line) this should be the largest text
    title_match = re.search(r'^([^\n]+)', text)
    if title_match:
        metadata['title'] = title_match.group(1).strip()

    # Extract publisher (assuming it contains "Publisher:" followed by the name)
    publisher_match = re.search(r'NEWSPAPER\s*([^\n]+)', text)
    if publisher_match:
        metadata['publisher'] = publisher_match.group(1).strip()

    # Extract date (assuming it contains "Date:" followed by the date)
    date_match = re.search(r'DATE\s*([^\n]+)', text)
    if date_match:
        metadata['date'] = date_match.group(1).strip()

    # Extract keywords (assuming they are listed after "Keywords:" or similar)
    keywords_match = re.search(r'Keywords:\s*([^\n]+)', text)
    if keywords_match:
        metadata['keywords'] = keywords_match.group(1).strip().split(',')

    return metadata

def analyze_text_files(output_folder):
    for filename in os.listdir(output_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(output_folder, filename), 'r') as txt_file:
                content = txt_file.read()
                metadata = extract_metadata_from_text(content)
                print(f"File: {filename}")
                print("Metadata:")
                for key, value in metadata.items():
                    print(f"{key}: {value}")
                print("\n")

if __name__ == "__main__":
    output_folder = "complete_images"

    analyze_text_files(output_folder)