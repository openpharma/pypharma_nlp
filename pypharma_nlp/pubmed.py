from Bio import Entrez
from Bio import Medline
from nltk.data import load
import matplotlib.pyplot as plt
import nltk
import os
import pandas as pd
import time


def _set_entrez_email():
    if "ENTREZ_EMAIL" not in os.environ.keys():
        print("Enter an e-mail for Entrez:")
        Entrez.email = input()
    else:
        if os.environ["ENTREZ_EMAIL"].strip() == "":
            print("Enter an e-mail for Entrez:")
            Entrez.email = input()
        else:
            Entrez.email = os.environ["ENTREZ_EMAIL"]


_set_entrez_email()


def publications_per_year(query=None, years=None, silent=False):
    
    """Get the number of publications in PubMed per year. Optionally given a 
    query."""
    
    if years == None:
        years = range(1970, 2019)
    counts = []
    if query == None:
        prefix = ""
    else:
        prefix = "(%s) AND " % query 

    # Show output table header
    if not silent:
        example_query = "%s(\"<YEAR>\"[Date - Publication])" % prefix
        print("Using query: '%s'" % example_query)
        print("\nyear count")
        print("----------")

    # Iterate over years
    for year in years:
        
        # Search pubmed
        query = "%s(\"%d\"[Date - Publication])" % (prefix, year)
        handle = Entrez.esearch(db="pubmed", term=query)
        time.sleep(1)
        record = Entrez.read(handle)
        count = int(record["Count"])

        # Output table row
        if not silent:
            print(year, count)
        
        # Update counts
        counts.append(count)
    return years, counts


def plot_publications_per_year(years, counts, about=None, 
    extra_footnote=None, output_path=None):

    """Plots a curve of publications per year as returned by 
    'get_publications_per_year'."""
    
    if about == None:
        title = "Number of publications* in PubMed per year"
    else:
        title = "Number of publications* about %s in PubMed per year" % about 
    footnote = "*According to the year of the 'Date - Publication' " + \
        "field of PubMed"
    if extra_footnote != None:
        footnote += "\n" + extra_footnote
    plt.title(title)
    plt.plot(years, counts, linestyle='-', marker='o')
    plt.figtext(0.01, 0.01, footnote, horizontalalignment="left", 
        fontsize="x-small")
    if output_path == None:
        plt.show()
    else:
        plt.savefig(output_path)


def get_search_results(query, max_results=None):
    
    """Get a list of PubMed results (PubMed IDs) for a query."""
    
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    time.sleep(1)
    search_results = Entrez.read(handle)
    pmids = search_results["IdList"]
    return pmids


def get_publications(query=None, pmids=None, max_results=None):
    
    """Get PubMed publications records for a query or list of ids."""

    # Get search results
    if query == None and pmids == None:
        raise ValueError("Either 'query' or 'pmids' must be not None.")
    elif query != None:
        pmids = get_search_results(query, max_results=max_results)
        
    # Return publication results
    handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", 
        retmode="text")
    time.sleep(1)
    records = Medline.parse(handle)
    return records
    
        
def get_publication_batches(query=None, pmids=None, max_results=None, 
    batch_size=64):
    
    """Get PubMed publication record batches for a query or list of ids."""

    # Get search results
    if query == None and pmids == None:
        raise ValueError("Either 'query' or 'pmids' must be not None.")
    elif query != None:
        pmids = get_search_results(query, max_results=max_results)
    pmids = pmids[:max_results]
        
    # Generate publication result batches
    cursor = 0
    while cursor < len(pmids):
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="medline", 
            retmode="text", retmax=batch_size, retstart=cursor)
        time.sleep(1)
        records = Medline.parse(handle)
        yield records
        cursor += batch_size


def get_publications_table(records):
    
    """Get a pandas table showing the data of a list of publication records as 
    returned by 'get_publications' or 'get_publication_batches'."""

    keys = ["PMID", "TI", "AU", "AB"]
    table_records = []
    for record in records:
        table_record = [record[k] for k in keys]
        table_record[2] = "; ".join(table_record[2])
        table_records.append(table_record)
    publications_table = pd.DataFrame.from_records(table_records, columns=keys)
    return publications_table


def get_publication_sentences(records, include_title=False, include_pmid=False, 
    spans=False, language="english"):
    
    """Get a generator of lists of sentences from a list of publication 
    records as returned by 'get_publications' or 'get_publication_batches'."""

    tokenizer = load('tokenizers/punkt/{0}.pickle'.format(language))
    for record in records:
        text = ""
        pmid = record["PMID"]
        if include_title:
            text += record["TI"] + "\n\n"
        text += record["AB"]
        offset = 0

        # With span tokenization
        if spans:
            tuples = []
            if include_pmid:
                tuples += [(pmid, (0, len(pmid)))]
                offset = 2 + len(pmid)
            tuples += [(text[b:e], (b + offset,e + offset)) for b, e in 
                tokenizer.span_tokenize(text)]
            sentences, spans = zip(*tuples)
            yield sentences, spans
            
        # Without span tokenization
        else:
            sentences = []
            if include_pmid:
                sentences = [pmid]
            sentences += tokenizer.tokenize(text)
            yield sentences
