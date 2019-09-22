from pypharma_nlp.biobert.data.ner_corpora import download_source_data
import os


# Data directories
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "data", "ner_corpora")

print("Downloading the ADE Corpus V2")
download_source_data(DATA_DIRECTORY)
