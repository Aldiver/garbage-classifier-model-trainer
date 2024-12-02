import os
import xml.etree.ElementTree as ET

def find_error_annotations(dataset_folder):
    # File to store the error filenames
    error_file_path = os.path.join(dataset_folder, "error_files.txt")
    error_files = []

    # Walk through all files and folders inside the dataset folder
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)

                try:
                    # Parse the XML file
                    tree = ET.parse(file_path)
                    root_element = tree.getroot()

                    # Look for <size> element
                    size_element = root_element.find("size")
                    if size_element is not None:
                        width = size_element.find("width").text

                        # Check if width is 0
                        if width == "0":
                            # Find the filename
                            filename_element = root_element.find("filename")
                            if filename_element is not None:
                                error_files.append(filename_element.text)
                except ET.ParseError:
                    print(f"Error parsing XML file: {file_path}")

    # Write all error files to error_files.txt
    with open(error_file_path, "w") as error_file:
        for error_filename in error_files:
            error_file.write(f"{error_filename}\n")

    print(f"Completed. Found {len(error_files)} error(s). Errors saved in {error_file_path}")

# Specify the dataset folder
dataset_folder = "datasets"
find_error_annotations(dataset_folder)
