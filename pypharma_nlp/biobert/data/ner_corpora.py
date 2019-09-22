from pypharma_nlp.drive import download_drive_file
from zipfile import ZipFile
import os


def download_source_data(data_directory, overwrite=False):
    
    """Downloads the source NER data from Biobert's Google drive account."""

    # If directory exists and not overwriting, stop
    if os.path.isdir(data_directory) and not overwrite:
        print("Found '%s', set 'overwrite' to True if you wish to overwrite it." % data_directory)
        return

    # Download BioBERT data
    os.makedirs(data_directory, exist_ok=True)
    file_id = "1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh"
    zip_path = os.path.join(data_directory, "NERdata.zip")
    download_drive_file(file_id, zip_path)

    # Unzip the file
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(data_directory)
    
    # Remove .zip file
    os.remove(zip_path)


def get_ner_examples(data_directory, task_name, subset):
    
    """Get ids, tokens and labels for named entity recognition.
    
    :param data_directory: The directory where the data was extracted.
    :param task_name: The task name. This is usually a subdirectory in the 
    data directory, e.g. 'BC5CDR-disease'.
    :param subset: The data subset. This is usually the name of file in the 
    task name subdirectory without the .tsv extension, e.g. 'train'.
    """
    
    # Read the data
    input_path = os.path.join(data_directory, task_name, subset + ".tsv")
    input_stream = open(input_path, "r")

    # Initialize buffers and iterate over lines
    sentence_ids, tokens, labels = [], [], []
    sentence_id = 1
    for line in input_stream.readlines():

        # New sentence started, yield the buffers
        if line.strip() == "":
            yield sentence_ids, tokens, labels
            sentence_ids, tokens, labels = [], [], []
            sentence_id += 1
        else:
            token, label = line.strip().split("\t")
            sentence_ids.append(sentence_id)
            tokens.append(token)
            labels.append(label)
        
    input_stream.close()
