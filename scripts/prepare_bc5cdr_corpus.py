from pypharma_nlp.biobert.data.ner_corpora import download_source_data
from pypharma_nlp.biobert.data.ner_corpora import get_ner_examples
import os


# Data directories
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "data", "ner_corpora")

print("Downloading the ADE Corpus V2")
download_source_data(DATA_DIRECTORY)

for sentence in get_ner_examples(DATA_DIRECTORY, "BC5CDR-disease", "train"):
    print(sentence)
