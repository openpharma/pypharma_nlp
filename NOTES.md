# Notes

## Datasets

### Available

- Datasets used in BioBERT paper
    - Paper link: https://arxiv.org/pdf/1901.08746.pdf
    - Code link: https://github.com/dmis-lab/biobert
    - Notes: The code seems to have instructions to download the data.
    - BioASQ: 
        - Link: http://bioasq.org/
        - Paper link: https://www.ncbi.nlm.nih.gov/pubmed/?term=An+overview+of+the+BIOASQ+large-scale+biomedical+semantic+indexing+and+question+answering+competition
        - Three sub-tasks:
        - Task A: Predict MeSH headings/subheadings (?) for new publications.
        - Task B: Provide answers with both relevant concepts, articles, snippets, rdf triples from designated resources, as well as exact and "ideal" answers.
            - Link (with info and downloads): http://participants-area.bioasq.org/general_information/Task7b/
            - Link (with info and downloads): http://participants-area.bioasq.org/Tasks/A/getData/
            - Training data link (4b): http://participants-area.bioasq.org/Tasks/4b/trainingDataset/
            - Test data link: http://participants-area.bioasq.org/Tasks/b/testData/
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
- Datasets used in NCBI_BERT (seems to outperform BioBert)
    
    - Paper link: https://arxiv.org/pdf/1906.05474.pdf
    - Code link (training)https://github.com/ncbi-nlp/NCBI_BERT
    - Code link (benchmark): https://github.com/ncbi-nlp/BLUE_Benchmark
    - Notes: 
        - It seems to be a modified version of the original BERT. 
        - It has scripts to run the training and prediction tasks (in the scripts folder).
        - It seems to have BioBERT checkpoints included, however their reported scores differ from the ones in the original paper.
    
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

## Methods

- Transformers
    - Video lecture by Vaswani (Transformers and Music Transformers): 
    https://www.youtube.com/watch?v=5vcj8kSwBCY&feature=youtu.be&t=2209
        - Constant path length between two positions.
        - Unbounded memory.
        - Trivial to paralellize.
        - Models Self-Similarity.
        - Relative attention and its effect on Music Transformers and Image 
          Transformers.
        - Non-autoregressive generation! (instead of word by word, decisions 
          made in a latent space, or iterative refinement, etc).
        - Transfer Learning BERT, etc, mesh tensorflow (larger models).
        - Other work: Universal Transformers (recurrence in depth), 
          Transformer XL (recurrence + self-attention, beyond fixed length 
          context), etc.
