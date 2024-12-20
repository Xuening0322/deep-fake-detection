import os
import requests
import tarfile
from requests.auth import HTTPBasicAuth
import argparse

# Use argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Download and extract the LRS2 dataset.")
parser.add_argument("--username", type=str, required=True, help="Username for dataset access")
parser.add_argument("--password", type=str, required=True, help="Password for dataset access")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save and extract the dataset")

# Parse command-line arguments
args = parser.parse_args()

# Get username, password, and output directory from arguments
USERNAME = args.username
PASSWORD = args.password
output_dir = args.output_dir

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the URL for the dataset
DATASET_URL = "https://thor.robots.ox.ac.uk/lip_reading/data2/lrs2_v1.tar"

# Path to save the downloaded TAR file
tar_file_path = os.path.join(output_dir, "lrs2_v1.tar")

# Function to download the dataset
def download_dataset(url, dest_path, username, password):
    try:
        print(f"Downloading {url} to {dest_path}")
        # Make a GET request with authentication
        with requests.get(url, auth=HTTPBasicAuth(username, password), stream=True) as response:
            response.raise_for_status()
            # Save the content to the specified file
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Download completed: {dest_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download dataset: {e}")

# Function to extract the TAR file
def extract_tar(file_path, extract_to):
    try:
        print(f"Extracting {file_path} to {extract_to}")
        # Open and extract the TAR file
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(path=extract_to)
        print(f"Extraction completed.")
    except tarfile.TarError as e:
        print(f"Failed to extract tar file: {e}")

# Download the LRS2 dataset
download_dataset(DATASET_URL, tar_file_path, USERNAME, PASSWORD)

# Extract the downloaded TAR file
extract_tar(tar_file_path, output_dir)

# Optionally delete the TAR file to save space
os.remove(tar_file_path)
print("TAR file removed after extraction.")
