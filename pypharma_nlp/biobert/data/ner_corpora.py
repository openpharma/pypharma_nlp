from pypharma.drive import download_drive_file
from zipfile import ZipFile
import os
import sys


def download_source_data(data_directory, overwrite=False):
    
    """Downloads the source NER data from Biobert's Google drive account."""

    # Download BioBERT data
    file_id = "1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh"
    zip_path = os.path.join(data_directory, "NERdata.zip")
    download_drive_file(file_id, zip_path)

    # Unzip the file
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(data_directory)
    
    # Remove .zip file
    os.remove(zip_path)
