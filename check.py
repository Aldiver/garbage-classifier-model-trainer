import os
import io
import PIL
import tensorflow as tf

# Define paths
train_dir = 'organized_data/train'
val_dir = 'organized_data/val'

# Helper function to check if an image file is a valid JPEG
def is_valid_jpeg(image_path):
    try:
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            return False
    except Exception as e:
        print(f"Error checking {image_path}: {e}")
        return False
    return True

# Check all .jpg files in the specified directory
def check_directory(directory):
    print(f"Checking directory: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                file_path = os.path.join(root, file)
                if not is_valid_jpeg(file_path):
                    print(f"Invalid JPEG file: {file_path}")

# Check both train and validation directories
check_directory(train_dir)
check_directory(val_dir)
