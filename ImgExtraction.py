import pandas as pd
import requests
from pathlib import Path
from urllib.parse import urlparse

# Function to download an image
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {url} to {save_path}")
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Load the CSV file
def extract_images_from_csv(csv_file_path, image_column_name, output_folder):
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure output directory exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Loop through each URL in the specified column
    for index, row in df.iterrows():
        image_url = row[image_column_name]
        if pd.notna(image_url):
            # Generate a valid file name from URL
            filename = urlparse(image_url).path.split('/')[-1]
            save_path = Path(output_folder) / filename
            # Download and save the image
            download_image(image_url, save_path)

# Example usage
if __name__ == "__main__":
    csv_file_path = 'images.csv'  # Path to your CSV file
    image_column_name = 'image_url'  # Name of the column with image URLs
    output_folder = 'downloaded_images'  # Folder to save the images
    extract_images_from_csv('dataset/train.csv', 'image_link', 'Train_Images')
