import os

def rename_files(output_folder):
    for filename in os.listdir(output_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(output_folder, filename), 'r') as txt_file:
                content = txt_file.read()
                # Add code to extract relevant information and create a new name
                new_filename = f"{extracted_info}_processed.txt"
                os.rename(os.path.join(output_folder, filename), os.path.join(output_folder, new_filename))

if __name__ == "__main__":
    output_folder = "complete_images"

    rename_files(output_folder)