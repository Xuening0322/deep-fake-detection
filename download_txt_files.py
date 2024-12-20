import os
import requests
from requests.auth import HTTPBasicAuth
import argparse

# Use argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Download LRS2 dataset text files.")
parser.add_argument("--username", type=str, required=True, help="Username for authentication")
parser.add_argument("--password", type=str, required=True, help="Password for authentication")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save downloaded text files")

# Parse command-line arguments
args = parser.parse_args()

# Get username, password, and output directory from arguments
USERNAME = args.username
PASSWORD = args.password
output_dir = args.output_dir

# Define the URLs for the text files
file_urls = {
    "train.txt": "https://thor.robots.ox.ac.uk/lip_reading/data2/train.txt",
    "val.txt": "https://thor.robots.ox.ac.uk/lip_reading/data2/val.txt",
    "test.txt": "https://thor.robots.ox.ac.uk/lip_reading/data2/test.txt",
}

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to download a file
def download_file(url, dest_path, username, password):
    try:
        print(f"Downloading {url} to {dest_path}")
        # Make a GET request with authentication
        with requests.get(url, auth=HTTPBasicAuth(username, password), stream=True) as response:
            response.raise_for_status()
            # Save the file content to the destination path
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded: {dest_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")

# Download each text file
for filename, url in file_urls.items():
    dest_path = os.path.join(output_dir, filename)
    download_file(url, dest_path, USERNAME, PASSWORD)

# Function to read and print the contents of a text file (optional)
def read_and_print_file(file_path):
    print(f"\nContents of {file_path}:")
    with open(file_path, "r") as f:
        for line in f:
            print(line.strip())  # Print each line without extra newline characters

# Optionally read and print the downloaded files
for filename in file_urls.keys():
    file_path = os.path.join(output_dir, filename)
    read_and_print_file(file_path)

print("\nAll files downloaded and processed.")
