import os
import cv2
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm  # Progress bar

# Define paths (replace these with your actual folder paths)
input_folder = 'D:/Amazon_ML/train_images'  # <--- Put your input folder path here
output_folder = 'D:/Amazon_ML/preprocessed'  # <--- Put your output folder path here

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to preprocess an image
def preprocess_image(filename):
    img_path = os.path.join(input_folder, filename)
    
    if os.path.isfile(img_path):
        try:
            # Load the image
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Image {filename} could not be loaded.")
                return  # Skip processing if image is not loaded properly

            # Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Normalize the image (scaling pixel values to the range 0-1)
            normalized_image = gray_image / 255.0
            
            # Resize the image (optional, you can adjust the size)
            resized_image = cv2.resize(normalized_image, (256, 256))  # Resize to 256x256, modify as needed
            
            # Save the preprocessed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, (resized_image * 255).astype(np.uint8))  # Save as uint8 image
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Function to process a batch of images in parallel
def process_images_in_parallel(image_filenames):
    with Pool(processes=8) as pool:  # Adjust 'processes' based on your CPU cores
        for _ in tqdm(pool.imap_unordered(preprocess_image, image_filenames), total=len(image_filenames)):
            pass  # The progress bar will update as images are processed

# Split the list of images into batches
def batch_process_images(batch_size=1000):
    # Filter only valid image formats
    valid_extensions = ['.jpg', '.jpeg', '.png']
    all_filenames = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in valid_extensions]

    num_batches = (len(all_filenames) + batch_size - 1) // batch_size  # Calculate total number of batches

    for i in range(0, len(all_filenames), batch_size):
        batch_filenames = all_filenames[i:i + batch_size]
        process_images_in_parallel(batch_filenames)
        print(f"Processed batch {i // batch_size + 1}/{num_batches}")

# Start batch processing
if __name__ == '__main__':
    batch_process_images()
    print(f"Preprocessing complete. Processed images saved in {output_folder}")
