from pypharma_nlp.data.hallmarks_of_cancer import download_source_data
from pypharma_nlp.data.hallmarks_of_cancer import get_classification_examples
import os


# Data directories
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "data", 
    "hallmarks_of_cancer")

download_source_data(DATA_DIRECTORY)
a = set()
for id, sentences, labels in get_classification_examples(DATA_DIRECTORY):
    for i in range(len(sentences)):
        #print(sentences[i][:50], labels[i])
        for x in labels[i]:
            a.add(x)
print(sorted(list(a)))
