from pypharma_nlp.drive import download_drive_file
import os
from zipfile import ZipFile


CHECKPOINT_ID_DICT = {
    "biobert_v1.1_pubmed_classification_ade" : "1awtztljFfKDaBG06VZgBQ6MwqmrtlsAv", 
    "biobert_v1.1_pubmed_ner_bc5cdr_disease" : "1_5Zqkj5ZLUOKUKEHFh8C18DSOTghwvph", 
    "biobert_v1.1_pubmed_qa_bioasq" : "13xPx35UPOQ5U7ObKikajqINGWtV35uU3", 
    "biobert_v1.1_pubmed_qa_squad" : "1F7_oMwY1bZSuUU6MGZuSj-Fz7Q5ovskD", 
    "biobert_v1.1_pubmed_re_gad" : "1K2p3dL62xzwOsmUIchjuqRgWOJklrRI9", 
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
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(checkpoint_directory)
    
    # Remove .tar file
    os.remove(zip_path)
