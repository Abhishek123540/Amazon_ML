import os
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# Enable mixed precision for performance boost
set_global_policy('mixed_float16')

# Disable TensorFlow logs to avoid verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define batch size for parallel processing
BATCH_SIZE = 32

def load_resnet50_model():
    # Load pre-trained ResNet50 model (no top layer, used for feature extraction)
    return ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the ResNet50 model once globally to avoid loading in every process
model = load_resnet50_model()

def preprocess_image(image_path):
    # Check if the image path is valid
    if not isinstance(image_path, str):
        raise ValueError(f"Invalid image path: {image_path}")
    
    # Load the grayscale image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if grayscale_image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Resize to 224x224 for ResNet50 input
    img = cv2.resize(grayscale_image, (224, 224))
    # Convert grayscale to 3-channel image by duplicating channels
    img = np.stack((img,) * 3, axis=-1)
    # Normalize image data to 0-1 range
    img = img.astype('float32') / 255.0
    return img

def extract_features_batch(image_paths):
    # Preprocess images
    img_arrays = np.array([preprocess_image(path) for path, _ in image_paths])
    
    # Extract features using ResNet50
    features = model.predict(img_arrays)
    
    # Flatten features for each image in the batch
    return features.reshape(features.shape[0], -1)

def process_image_batch(args):
    input_image_paths, output_feature_paths = args
    
    # Extract features for the batch using the pre-loaded model
    features = extract_features_batch(input_image_paths)
    
    # Save the features for each image in the batch
    for i, output_feature_path in enumerate(output_feature_paths):
        np.save(output_feature_path, features[i])
        print(f"Extracted and saved features to {output_feature_path}")

def create_batches(image_paths, batch_size):
    # Group image paths into batches
    return [
        (image_paths[i:i + batch_size], [output[1] for output in image_paths[i:i + batch_size]])
        for i in range(0, len(image_paths), batch_size)
    ]

def extract_features_from_images_multiprocessing(input_folder, output_folder, num_processes=4):
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a list of image paths and corresponding output feature paths
    image_paths = [
        (os.path.join(input_folder, filename), os.path.join(output_folder, filename.split('.')[0] + '.npy'))
        for filename in os.listdir(input_folder)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    
    # Create batches
    batches = create_batches(image_paths, BATCH_SIZE)
    
    # Use multiprocessing to parallelize feature extraction in batches
    with Pool(processes=num_processes) as pool:
        pool.map(process_image_batch, batches)

# Example usage
if __name__ == "__main__":
    input_folder = r"D:/Amazon_ML/preprocessed"  # Folder containing grayscale images
    output_folder = r"D:/Amazon_ML/output"  # Folder to save extracted features
    
    # Extract features from all images in the folder using multiple CPU processes
    extract_features_from_images_multiprocessing(input_folder, output_folder, num_processes=8)
