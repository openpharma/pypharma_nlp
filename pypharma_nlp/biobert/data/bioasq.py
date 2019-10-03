from pypharma_nlp.drive import download_drive_file
from zipfile import ZipFile
import json
import os
import shutil


def download_source_data(data_directory, overwrite=False):
    
    """Downloads the source BioASQ data from Biobert's Google drive account."""

    # If directory exists and not overwriting, stop
    if os.path.isdir(data_directory) and not overwrite:
        print("Found '%s', set 'overwrite' to True if you wish to overwrite it." % data_directory)
        return

    # Download BioBERT data
    os.makedirs(data_directory, exist_ok=True)
    file_id = "19ft5q44W4SuptJgTwR84xZjsHg1jvjSZ"
    zip_path = os.path.join(data_directory, "QA.zip")
    download_drive_file(file_id, zip_path)

    # Unzip the file
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(data_directory)

    # Move files to root data directory and remove sub-directory
    sub_directory = os.path.join(data_directory, "BioASQ")
    for path in os.listdir(sub_directory):
        shutil.move(os.path.join(sub_directory, path), data_directory)
    os.rmdir(sub_directory)

    # Remove .zip file
    os.remove(zip_path)


def get_qa_examples(data_directory, subset, task):

    """Gets the BioASQ question answering examples.
    
    :param data_directory The directory where the BioASQ data is.
    :param subset The data subset (train, or test).
    :param task The BioASQ task (e.g. 6b).
    """

    input_path = os.path.join(data_directory, "BioASQ-%s-factoid-%s.json" % \
        (subset, task))
    data = json.load(open(input_path, "r"))
    ids, contexts, questions, answers = [], [], [], []
    for item in data["data"][0]["paragraphs"]:
        context = item["context"]
        current_ids, current_questions, current_answers = [], [], []
        for qa in item["qas"]:
            id = qa["id"]
            question = qa["question"]
            current_ids += [id] * len(qa["answers"])
            current_questions += [question] * len(qa["answers"])
            current_answers += [a["text"] for a in qa["answers"]]
        ids += current_ids
        contexts = [context] * len(current_questions)
        questions += current_questions
        answers += current_answers
    return ids, contexts, questions, answers
