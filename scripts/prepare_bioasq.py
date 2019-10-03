from pypharma_nlp.biobert.data.bioasq import download_source_data
from pypharma_nlp.biobert.data.bioasq import get_qa_examples
import os


# Data directories
SCRIPT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "data", "bioasq")

print("Downloading the BioASQ")
download_source_data(DATA_DIRECTORY)

print("Processing the BioASQ")
ids, contexts, questions, answers = get_qa_examples(DATA_DIRECTORY, "train", 
    "6b")
print(ids, contexts, questions, answers)
