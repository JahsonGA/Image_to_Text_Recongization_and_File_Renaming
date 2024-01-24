import os
import cv2
import pytesseract
from shutil import move

def process_images(input_folder, output_folder, review_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            image_path = os.path.join(input_folder, filename)
            try:
                image = cv2.imread(image_path)
                text = pytesseract.image_to_string(image)
                if text:
                    with open(os.path.join(output_folder, f"{filename}.txt"), 'w') as txt_file:
                        txt_file.write(text)
                    move(image_path, os.path.join(output_folder, filename))
                else:
                    move(image_path, os.path.join(review_folder, filename))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                move(image_path, os.path.join(review_folder, filename))

if __name__ == "__main__":
    input_folder = "unnamed_file"
    output_folder = "complete_images"
    review_folder = "manual_review_images"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(review_folder):
        os.makedirs(review_folder)

    process_images(input_folder, output_folder, review_folder)