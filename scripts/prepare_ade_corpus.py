from pypharma_nlp.data.ade_corpus import download_source_data
from pypharma_nlp.data.ade_corpus import get_classification_examples
import os


# Data directories
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "data", "ade_corpus")

print("Downloading the ADE Corpus V2")
download_source_data(DATA_DIRECTORY)

print("Processing the ADE Corpus V2")
counts = {}
for pmid, sentences, labels in get_classification_examples(DATA_DIRECTORY):
    for i in range(len(sentences)):
        if labels[i] not in counts.keys():
            counts[labels[i]] = 0
        counts[labels[i]] += 1

print(counts)
