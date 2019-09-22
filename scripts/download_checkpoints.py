from pypharma_nlp.biobert.checkpoints.base import CHECKPOINT_ID_DICT
from pypharma_nlp.biobert.checkpoints.base import download_checkpoint
import os


if __name__ == "__main__":
    for checkpoint in CHECKPOINT_ID_DICT.keys():
        checkpoint_directory = os.path.join("models", "biobert")
        download_checkpoint(checkpoint_directory, checkpoint=checkpoint)
