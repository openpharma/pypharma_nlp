# Notes

## Datasets

### Available

- Datasets used in BioBERT paper
    - BioASQ: 
        - Link: http://bioasq.org/
        - Paper link: https://www.ncbi.nlm.nih.gov/pubmed/?term=An+overview+of+the+BIOASQ+large-scale+biomedical+semantic+indexing+and+question+answering+competition
        - Three sub-tasks:
        - Task A: Predict MeSH headings/subheadings (?) for new publications.
        - Task B: Provide answers with both relevant concepts, articles, snippets, rdf triples from designated resources, as well as exact and "ideal" answers.
            - Link (with downloads): http://participants-area.bioasq.org/general_information/Task7b/
        - Task C: Classify IBECS and LILACS documents (similar to Task A but in spanish).

    - BioCreative BC5CDR and ChemProt:
        - General Link: https://biocreative.bioinformatics.udel.edu/tasks/
        - BC5CDR 
            - Link: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/
            - Paper link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/
            - Papers with code link: https://paperswithcode.com/sota/named-entity-recognition-ner-on-bc5cdr
        - ChemProt:
            - Link: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/
            - Paper link: https://www.semanticscholar.org/paper/Overview-of-the-BioCreative-VI-chemical-protein-Krallinger-Rabal/eed781f498b563df5a9e8a241c67d63dd1d92ad5
            - Papers with code link: https://paperswithcode.com/sota/relation-extraction-on-chemprot

    - EU-ADR (adverse drug reactions):
        - Paper link: https://www.sciencedirect.com/science/article/pii/S1532046412000573?via%3Dihub
        - Data link: https://biosemantics.org/downloads/euadr.tgz

    - GAD (genetic associations):
        - Paper link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4466840/#!po=58.3333
        - Link: http://ibi.imim.es/befree/#corpora
        - Data link: https://geneticassociationdb.nih.gov/data.zip
     
- BioNLP 2019
    - MEDIQA: 
        - Link: https://sites.google.com/view/mediqa2019
        - Three sub-tasks: inference relations, entailment, and ranking answers to questions from CHiQA (using one of both of the earlier tasks).
        - For NLI there is a simple to use baseline: https://github.com/jgc128/mednli_baseline.
        - The CHiQA system does not appear to be open source.
- BioCaddie
- Datasets used in Peng "Transfer Learning..." (seems to outperform BioBert)
    
    - Code link: https://github.com/ncbi-nlp/BLUE_Benchmark
    
    - BIOSSES: 
        - Link: http://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html
        - Paper link: https://www.ncbi.nlm.nih.gov/pubmed/28881973
        - Data link: http://tabilab.cmpe.boun.edu.tr/BIOSSES/Downloads/BIOSSES-Dataset.rar

    - Hallmarks of Cancers corpus:
        - Link: https://www.cl.cam.ac.uk/~sb895/HoC.html
        - Paper link: https://academic.oup.com/bioinformatics/article/32/3/432/1743783
        - Data link: https://www.cl.cam.ac.uk/~sb895/HoC_Preprocessed.zip

### Selected Datasets for Workshop

- For classification (automated tagging)
    - Hallmark of Cancers (HoC)
    - BioASQ Task A

- For NER
    - BC5CDR

- For relation extraction:
    - CHEMPROT
    - EU-ADR

- Sentence similarity
    - BIOSSES
    - MedSTS

- For question answering:
    - BioASQ Task B
