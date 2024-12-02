import os
import csv
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image  # For image conversion
import xml.etree.ElementTree as ET  # For XML manipulation
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
import tensorflow as tf
assert tf.__version__.startswith('2')

# Define paths
dataset_path = 'datasets'
train_dir = 'organized_data/train'
val_dir = 'organized_data/val'
csv_file = 'datasets.csv'  # CSV file to store paths

# Categories and subcategories (classes)
categories = [
    'Metal', 'Paper', 'Plastic'
]

# categories = [
#     'chopstick', 'leaf', 'toothpick', 'Wooden Utensils', 'Juice Box', 'Paper Packages',
#     'cardboard', 'glass', 'metal', 'paper', 'plastic', 'bandaid', 'diaper', 'milkbox', 'napkin',
#     'pen', 'plasticene', 'rag', 'toothbrush', 'toothpastetube'
# ]

# Subcategories for each category
subcategories = {
    'Biodegradeable': ['Paper'],
    'Recyclable': ['Metal'],
    'Residual': ['Plastic']
}

# Helper function to convert PNG to JPEG and update XML
def convert_to_jpeg(image_path):
    if image_path.endswith('.png'):
        img_name = os.path.splitext(image_path)[0]
        jpg_image_path = f"{img_name}.jpg"

        # Convert PNG to JPEG
        with Image.open(image_path) as img:
            rgb_im = img.convert('RGB')
            rgb_im.save(jpg_image_path)
            print(f"Converted {image_path} to {jpg_image_path}")

        # Update corresponding XML file
        xml_path = image_path.replace('.png', '.xml')
        if os.path.exists(xml_path):
            update_xml_file(xml_path, jpg_image_path)

        return jpg_image_path
    return image_path

# Helper function to update XML file
def update_xml_file(xml_path, new_image_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update <filename> element
    filename_element = root.find('filename')
    if filename_element is not None:
        new_filename = os.path.basename(new_image_path)
        filename_element.text = new_filename

    # Update <path> element
    path_element = root.find('path')
    if path_element is not None:
        new_path = os.path.abspath(new_image_path)
        path_element.text = new_path

    # Save the updated XML file
    tree.write(xml_path)
    print(f"Updated XML file {xml_path} with new path {new_image_path}")

# Helper function to replace .jpeg with .jpg in filenames and XML paths
def replace_jpeg_with_jpg(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpeg'):
                old_file_path = os.path.join(root, file)
                new_file_path = old_file_path.replace('.jpeg', '.jpg')

                # Rename the image file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed {old_file_path} to {new_file_path}")

                # Update corresponding XML file
                xml_file = file.replace('.jpeg', '.xml')
                old_xml_path = os.path.join(root, xml_file)
                new_xml_path = old_xml_path.replace('.jpeg', '.jpg')

                if os.path.exists(old_xml_path):
                    update_xml_file(old_xml_path, new_file_path)

                    # Rename the XML file
                    os.rename(old_xml_path, new_xml_path)
                    print(f"Renamed XML file {old_xml_path} to {new_xml_path}")

# Helper function to collect data and ensure JPEG format
def collect_data(category, subcategory):
    image_dir = os.path.join(dataset_path, category, subcategory)
    images = [img for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png'))]

    # Check if both image and its corresponding XML file exist
    data = []
    for img in images:
        xml_file = img.replace('.jpg', '.xml').replace('.png', '.xml')
        xml_path = os.path.join(image_dir, xml_file)

        if os.path.exists(xml_path):  # Only add if the XML file exists
            image_path = os.path.join(image_dir, img)
            jpeg_image_path = convert_to_jpeg(image_path)
            jpeg_image_name = os.path.basename(jpeg_image_path)
            data.append((jpeg_image_name, xml_file))
        else:
            print(f"Skipping {img} as corresponding {xml_file} is missing.")
    return data

# Helper function to split and organize data into train and validation directories
def organize_data(category, subcategory, train_dir, val_dir, test_size=0.2):
    data = collect_data(category, subcategory)
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)

    # Create directories for train and validation sets
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Copy files to appropriate directories
    for dataset, dir_path in [(train_data, train_dir), (val_data, val_dir)]:
        for image, annotation in dataset:
            shutil.copy(os.path.join(dataset_path, category, subcategory, image), dir_path)
            shutil.copy(os.path.join(dataset_path, category, subcategory, annotation), dir_path)
    return train_data, val_data

# Replace .jpeg with .jpg in filenames and XML paths
replace_jpeg_with_jpg(dataset_path)

# Create the CSV file and store file paths for train and validation sets
with open(csv_file, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File Type', 'Category', 'Subcategory', 'Image', 'Annotation'])  # CSV header

    # Organize data for each category and subcategory and log the data
    for category in subcategories:
        for subcategory in subcategories[category]:
            train_data, val_data = organize_data(category, subcategory, train_dir, val_dir)

            # Write train data to CSV
            for image, annotation in train_data:
                csv_writer.writerow(['Train', category, subcategory, image, annotation])

            # Write validation data to CSV
            for image, annotation in val_data:
                csv_writer.writerow(['Validation', category, subcategory, image, annotation])

# Load the data using DataLoader
train_data = object_detector.DataLoader.from_pascal_voc(
    train_dir, train_dir, categories)

val_data = object_detector.DataLoader.from_pascal_voc(
    val_dir, val_dir, categories)

# Load the EfficientDet Lite 2 model specification
spec = model_spec.get('efficientdet_lite0')

# Create and train the object detection model
model = object_detector.create(
    train_data,
    model_spec=spec,
    batch_size=4,
    train_whole_model=True,
    epochs=20,
    validation_data=val_data
)

# Evaluate the model
model.evaluate(val_data)

# Export the model
model.export(export_dir='.', tflite_filename='garbage_classifier_model.tflite')

# Evaluate the exported TFLite model
model.evaluate_tflite('garbage_classifier_model.tflite', val_data)
