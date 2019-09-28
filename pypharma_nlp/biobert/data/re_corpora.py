from pypharma_nlp.drive import download_drive_file
from zipfile import ZipFile
import os


def download_source_data(data_directory, overwrite=False):
    
    """Downloads the source RE data from Biobert's Google drive account."""

    # If directory exists and not overwriting, stop
    if os.path.isdir(data_directory) and not overwrite:
        print("Found '%s', set 'overwrite' to True if you wish to overwrite it." % data_directory)
        return

    # Download BioBERT data
    os.makedirs(data_directory, exist_ok=True)
    file_id = "1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw"
    zip_path = os.path.join(data_directory, "REdata.zip")
    download_drive_file(file_id, zip_path)

    # Unzip the file
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(data_directory)
    
    # Remove .zip file
    os.remove(zip_path)


def get_re_examples(data_directory, task_name, fold, subset):
    
    """Get ids, tokens and labels for relation extraction.
    
    :param data_directory: The directory where the data was extracted.
    :param task_name: The task name. This is usually a subdirectory in the 
    data directory, e.g. 'GAD'.
    :param fold: The cross validation fold (1-10).
    :param subset: The data subset. This is usually the name of file in the 
    task name subdirectory without the .tsv extension, e.g. 'train'.
    """
    
    # Read the data
    input_path = os.path.join(data_directory, task_name, fold, subset + ".tsv")
    input_stream = open(input_path, "r")
    count = 1
    sentence_ids, sentences, labels = [], [], []

    # We skip a line on the test data, since it has headers
    if subset == "test":
        input_stream.readline()
        
    for line in input_stream.readlines():

        # Training and test data have slightly different formats
        if subset == "train":
            sentence, label = line.strip().split("\t")
        elif subset == "test":
            _, sentence, label = line.strip().split("\t")
        
        sentence_ids.append(count)
        sentences.append(sentence)
        labels.append(label)
        count += 1
    input_stream.close()
    return sentence_ids, sentences, labels
