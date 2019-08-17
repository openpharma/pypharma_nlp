from Bio import Entrez
import matplotlib.pyplot as plt
import os
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


def publications_per_year(query=None, years=None, silent=False):
    
    # Setup
    _set_entrez_email()
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
