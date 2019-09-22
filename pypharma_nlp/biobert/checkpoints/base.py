from pypharma_nlp.drive import download_drive_file
import os
import tarfile


CHECKPOINT_ID_DICT = {
    "biobert_v1.1_pubmed" : "1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD", 
    "biobert_v1.0_pubmed" : "17j6pSKZt5TtJ8oQCDNIwlSZ0q5w7NNBg", 
    "biobert_v1.0_pmc" : "1LiAJklso-DCAJmBekRTVEvqUOfm0a9fX", 
    "biobert_v1.0_pubmed_pmc" : "1jGUu2dWB1RaeXmezeJmdiPKQp3ZCmNb7", 
}


def download_checkpoint(checkpoint_directory, overwrite=False, 
    checkpoint="biobert_v1.1_pubmed"):
    
    """Download a BioBERT checkpoint."""

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
    tar_path = os.path.join(checkpoint_directory, "%s.tar.gz" % checkpoint)
    download_drive_file(file_id, tar_path)

    # Untar the file
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(checkpoint_directory)
    
    # Remove .tar file
    os.remove(tar_path)
