from nltk.tokenize.treebank import TreebankWordDetokenizer
from zipfile import ZipFile
import os
import warnings
import wget


def download_source_data(data_directory, overwrite=False):
    
    """Downloads the source data of the Hallmarks of Cancer (HoC) dataset from 
    https://www.cl.cam.ac.uk/~sb895/HoCCorpus.zip"""

    if os.path.isdir(data_directory) and not overwrite:
        warnings.warn("Found '%s', skipping. Use 'overwrite=True' if you wish to overwrite a file." % data_directory)
    else:
        os.makedirs(data_directory, exist_ok=True)
        url = "https://www.cl.cam.ac.uk/~sb895/HoCCorpus.zip"
        filename = url.split("/")[-1]
        output_path = os.path.join(data_directory, filename)
        wget.download(url, out=output_path)
        with ZipFile(output_path, "r") as z:
            z.extractall(path=data_directory)
            shutil.move(data_directory, data_directory + ".old")
            subdirectory = os.path.join(data_directory + ".old", "HoCCorpus")
            shutil.copytree(subdirectory, data_directory)
            shutil.rmtree(data_directory + ".old")
        if os.path.isfile(output_path):
            os.remove(output_path)


def get_classification_examples(data_directory):
    
    """Get ids, sentences and labels for text classification."""

    detokenizer = TreebankWordDetokenizer()
    filenames = os.listdir(data_directory)
    for filename in filenames:
        pmid = filename[:-4]
        input_path = os.path.join(data_directory, filename)
        with open(input_path, "r") as f:
            sentences = []
            labels_list = []
            for line in f.readlines():
                tokens = line.split("\t")[0].split(" ")
                sentence = detokenizer.detokenize(tokens)
                sentences.append(sentence)
                labels_string = line.split("\t")[1][1:-2].replace("'", "")
                labels = []
                if labels_string != "":
                    labels = labels_string.split(", ")
                labels_list.append(labels)
            yield pmid, sentences, labels_list
