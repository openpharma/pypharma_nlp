from pypharma_nlp.pubmed import get_publication_batches
from pypharma_nlp.pubmed import get_publication_sentences
import os
import pandas as pd
import wget


def download_source_data(data_directory, overwrite=False):
    
    """Download the source data of the ADE Corpus V2 from Trung Huynh's 
    repository: https://github.com/trunghlt/AdverseDrugReaction"""

    urls = [
        "https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/ADE-NEG.txt", 
        "https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/DRUG-AE.rel", 
        "https://raw.githubusercontent.com/trunghlt/AdverseDrugReaction/master/ADE-Corpus-V2/DRUG-DOSE.rel"
    ]
    os.makedirs(data_directory, exist_ok=True)
    for url in urls:
        filename = url.split("/")[-1]
        output_path = os.path.join(data_directory, filename)
        if os.path.isfile(output_path) and not overwrite:
            print("Found '%s', skipping. Use 'overwrite=True' if you wish to overwrite a file." % output_path)
            continue
        wget.download(url, out=output_path)


def read_drug_ae_data(data_directory):
    
    """Read the DRUG-AE relationship data into a data frame."""
    
    input_path = os.path.join(data_directory, "DRUG-AE.rel")
    drug_ae_data = pd.read_csv(input_path, delimiter="|", header=None, 
        names=[
            "PMID", 
            "SENTENCE", 
            "CONDITION", 
            "CONDITION_START", 
            "CONDITION_END", 
            "DRUG", 
            "DRUG_START", 
            "DRUG_END", 
        ])
    drug_ae_data["PMID"] = drug_ae_data["PMID"].apply(str)
    drug_ae_data.sort_values(by=["PMID", "CONDITION_START"], inplace=True)
    return drug_ae_data


def read_ade_neg_pmids(data_directory):
    
    """Read the PMIDs of the ADE-NEG data into a set."""
    
    pmids = set()
    input_path = os.path.join(data_directory, "ADE-NEG.txt")
    with open(input_path, "r") as f:
        for line in f.readlines():
            pmid = line.split(" ")[0]
            pmids.add(pmid)
    return pmids


def _get_span_dict(drug_ae_data):
    condition_span_dict = {}
    for i in range(drug_ae_data.shape[0]):
        pmid = drug_ae_data["PMID"][i]
        condition_span = (
            drug_ae_data["CONDITION_START"][i], 
            drug_ae_data["CONDITION_END"][i], 
        )
        if pmid not in condition_span_dict.keys():
            condition_span_dict[pmid] = []
        condition_span_dict[pmid].append(condition_span)
    return condition_span_dict


def _get_labels(sentences, sentence_spans, condition_span_dict):
    pmid = sentences[0]
    labels = []
    if pmid in condition_span_dict.keys():
        condition_spans = condition_span_dict[pmid]
        i = 1
        j = 0
        condition_span = condition_spans[j]
        while i < len(sentence_spans):
            while sentence_spans[i][1] < condition_span[0]:
                labels.append("Neg")
                i += 1
            labels.append("AE")
            while condition_span[1] <= sentence_spans[i][0]:
                j += 1
                if j >= len(condition_spans):
                    break
                condition_span = condition_spans[j]
            if j >= len(condition_spans):
                break
            i += 1
    if len(labels) < len(sentences) - 1:
        labels += ["Neg"] * (len(sentences) - len(labels) - 1)
    return labels


def get_classification_examples(data_directory, batch_size=10, max_results=10):
    
    """Get a list of sentences and labels for text classification."""
    
    drug_ae_data = read_drug_ae_data(data_directory)
    condition_span_dict = _get_span_dict(drug_ae_data)
    pmids = set(condition_span_dict.keys())
    neg_pmids = read_ade_neg_pmids(data_directory)
    pmids = list(pmids.union(neg_pmids))
    for batch in get_publication_batches(pmids=pmids, batch_size=batch_size, 
        max_results=max_results):
        documents = get_publication_sentences(batch, include_title=True, 
            include_pmid=True, spans=True)
        for sentences, sentence_spans in documents:
            labels = _get_labels(sentences, sentence_spans, condition_span_dict)
            yield sentences[1:], labels
