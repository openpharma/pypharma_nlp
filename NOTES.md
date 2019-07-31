# Notes

## Datasets

### Available

- Datasets used in BioBERT paper

    - BioASQ: 
       - Link: http://bioasq.org/
       - Three sub-tasks:
        - Task A: Predict MeSH headings/subheadings (?) for new publications.
        - Task B: Provide answers with both relevant concepts, articles, snippets, rdf triples from designated resources, as well as exact and "ideal" answers.
        - Task C: Classify IBECS and LILACS documents (similar to Task A but in spanish).

    - BioCreative BC5CDR and CHEMPROT:
        - General Link: https://biocreative.bioinformatics.udel.edu/tasks/
        - BC5CDR 
            - Link: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/
            - Papers with code link: https://paperswithcode.com/sota/named-entity-recognition-ner-on-bc5cdr
        - CHEMPROT:
            - Link: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/
            - https://paperswithcode.com/sota/relation-extraction-on-chemprot
     
- BioNLP 2019
    - MEDIQA: 
        - Link: https://sites.google.com/view/mediqa2019
        - Three sub-tasks: inference relations, entailment, and ranking answers to questions from CHiQA (using one of both of the earlier tasks).
        - For NLI there is a simple to use baseline: https://github.com/jgc128/mednli_baseline.
        - The CHiQA system does not appear to be open source.
- BioCaddie
- Datasets used in Peng "Transfer Learning..." (seems to outperform BioBert)

### Selected Datasets for Workshop

- For classification
    - See BioASQ Task A above.

- For NER
    - BC5CDR, see above.

- For relation extraction:
    - CHEMPROT, see above.
