from pypharma_nlp.biobert.data.re_corpora import download_source_data
from pypharma_nlp.biobert.data.re_corpora import get_re_examples
import os


# Data directories
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "data", "re_corpora")

print("Downloading the ADE Corpus V2")
download_source_data(DATA_DIRECTORY)

for sentence in get_re_examples(DATA_DIRECTORY, "GAD", "1", "train"):
    print(sentence)
