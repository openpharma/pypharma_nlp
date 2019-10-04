from pypharma_nlp.drive import download_drive_file
import os
from zipfile import ZipFile


CHECKPOINT_ID_DICT = {
    "classification" : "15JNbxQG1ffnLQ3tZlLoVjaRcGljcVLZ1", 
    "ner" : "1LVUSPaniDpecVUpYmDeFCKXWSXK_WsQH", 
}


def download_checkpoint(checkpoint_directory, checkpoint, 
    overwrite=False):
    
    """Download a PyPharma NLP checkpoint."""

    # If directory exists and not overwriting, stop
    checkpoint_subdirectory = os.path.join(checkpoint_directory, 
        checkpoint)
    if os.path.isdir(checkpoint_subdirectory) and not overwrite:
        print("Found '%s', set 'overwrite' to True if you wish to overwrite it." % checkpoint_subdirectory)
        return

    # Check if checkpoint exists
    checkpoints = CHECKPOINT_ID_DICT.keys()
    if checkpoint not in checkpoints:
        raise ValueError("Checkpoint must be one of: %s" % \
            ", ".join(checkpoints))
    file_id = CHECKPOINT_ID_DICT[checkpoint]
    
    # Download BioBERT data
    os.makedirs(checkpoint_directory, exist_ok=True)
    zip_path = os.path.join(checkpoint_directory, "%s.zip" % checkpoint)
    download_drive_file(file_id, zip_path)

    # Unzip the file
    with ZipFile.open(zip_path, "r") as zf:
        zf.extractall(checkpoint_directory)
    
    # Remove .tar file
    os.remove(zip_path)
