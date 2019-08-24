% PyPharma NLP Workshop 2019: \
  Introduction to Biomedical NLP
% Diego Saldana, Data Scientist at Roche
% November 2019


# Why Biomedical NLP?


## The Information Flood (1/3)

* Most of the information out there is in the form of natural language: 
scientific papers, clinical notes, social media, textbooks, lectures, 
websites.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/num_publications_year.pdf}
\end{figure}


## The Information Flood (2/3)

* Most of the information out there is in the form of natural language: 
scientific papers, clinical notes, social media, textbooks, lectures, 
websites.

\begin{figure}
\includegraphics[height=0.5\textheight]{figures/wikipedia_timeline.png}
\end{figure}


## The Information Flood (3/3)

* Most of the information out there is in the form of natural language: 
scientific papers, clinical notes, social media, textbooks, lectures, 
websites.
* This information is potentially very useful but cannot readily be used 
programmatically and stored in databases, searched, or analyzed.
* As a result this valuable information is "locked into a vault" until a human 
reads it, structures it and puts it into some database.
* And even when that happens, the scope in which the data can be used is 
usually limited and chosen by the extractors.
* How can machines help?


## Humans vs. Machines (1/2)

* Machines and humans have different strengths and weaknesses when processing 
text.

\begin{figure}
\includegraphics[height=0.7\textheight]{figures/ade_corpus_iaa.png}
\end{figure}


## Humans vs. Machines (2/2)

* Machines and humans have different strengths and weaknesses when processing 
text.
* Machines in particular are capable of processing vast amounts of text in a 
very short period of time in a very consistent way and performing simple 
tasks.
* Humans are take much more time to process text and are less consistent, 
however they are capable of much more complex reasoning and understanding.


## Humans vs. Machines (3/2)

What are some examples of tasks can computers perform well in 2019?

* Categorizing documents (e.g. automatically assigning MeSH headings to 
PubMed abstracts)
* Extracting entities from text (e.g. extracting Drugs, Diseases from 
PubMed abstracts)
* Extracting relations from text (e.g. extracting Adverse Events from 
PubMed abstracts)
* Answering simple questions based on a small amount of context (e.g. 
"Which drug should be used as an antidote in benzodiazepine overdose?")


# Some Natural Language Processing Tasks


## Language Modelling

A language model assigns probabilities to sequences of tokens, where tokens 
$t$ can be words, characters, sub-words, etc: 

$P\Big(t_{1}, t_{2}, t_{3}, ..., t_{N}\Big)$.

Take four sentences:

* ``The dog ran after the cat.''
* ``The dog ran after the tiger.''
* ``The stone ran after the tiger.''
* ``Tiger stone the after ran.''

Clearly, each subsequent sentence is less probable than the next. A good 
language model should assign probabilities to these sentences accordingly.


## Document Classification (1/2)

A document classifier assigns one or more class labels to a document.

Examples of document classification include:

* Predicting MeSH headings for PubMed abstracts.
* Annotating PubMed abstracts according to the Hallmarks of Cancer (HoC).
* Classifying sentences as having mentions of Adverse Drug Reactions (ADRs) or 
not.


## Document Classification (2/2)

A document classifier assigns one or more class labels to a document.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/hoc_annotation.jpeg}
\end{figure}


## Named Entity Recognition

A Named Entity Recognizer extracts entities from a document. 

* Examples of potential named entities include: drugs, diseases, genes, 
mutations, proteins, etc.
* One can extract the entities themselves as well as the boundaries. That is, 
the start and the end of the entity mention in the text.
* One can also subsequently perform Named Entity Resolution: Mapping the 
extracted entity to a concept in a standardized vocabulary.


## Relation Extraction

A Relation Extractor extracts two or more entities and a relationship between 
them. Examples of potential relations to extract include:

* A drug inducing an adverse reaction.
* A gene mutation inducing resistance to a drug.
* A gene regulating a biological pathway.
* A drug targetting a protein.
* A protein interacting with another protein.
* etc

## Question Answering

A Question Answering system provides an answer to a question given some 
context. That is, a set of documents. An example question would be:

\textbf{Context:} Orteronel is an investigational, partially selective inhibitor of CYP 17,20-lyase in the androgen signalling pathway, a validated therapeutic target for metastatic castration-resistant prostate cancer. ...

--------------------------------------
\textbf{Question:} Orteronel was developed for treatment of which cancer?
\textbf{Answer:} castration-resistant prostate cancer
--------------------------------------


# Some Highlights in the History of NLP

## Bag-Of-Words models

Bag-Of-Words (BOW) models ignore context and ordering of the words in a 
sentence and model them as an unordered collection of words [@bow].

* Often the words are pre-processed: lowercasing, stemming, 
removing stop words, tf-idf, etc.
* Early uses of bags of words were notably in spam filtering.
* We can build a word-document-matrix with documents as rows and words as 
columns. 
* Note that such a matrix is very sparse.

-------------------------------------------------------
\textbf{Sentence:} The dog was barking at the other dog. 
\textbf{BOW representation:} dog: 2, bark: 1, other: 1, all other words: 0
-------------------------------------------------------


## Latent Semantic Analysis

Applying Singular Value Decomposition to a Word-Document-Matrix is referred to 
as Latent Semantic Analysis [@lsa].

* We obtain latent variables representing a space in which words having 
  similar meanings are closer to each other than words having very distant 
  meanings.
* The model can thereby deal with synonyms, antonyms, singular-plural 
  forms of words, etc.
* Similar documents are also close to each other in latent space.
* An early method for distributional semantics.
* It's also a form of dimensionality reduction.

-------------------------------------------------------
\textbf{Sentence:} The dog was barking at the other dog. 
\textbf{BOW representation:} dog: 2, bark: 1, other: 1, all other words: 0
-------------------------------------------------------


## Latent Dirichlet Allocation (1/2)

Latent Dirichlet Allocation [@lda] is a bayesian approach that models the 
document generating process as a probabilistic graphical model. We have:

* A distribution of words over topics.
* A distribution of topics over documents.
* Each document is a collection of topic-word pairs drawn from these 
  distributions.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/lda.png}
\end{figure}


## Latent Dirichlet Allocation (2/2)

Latent Dirichlet Allocation [@lda] is a bayesian approach that models the 
document generating process as a probabilistic graphical model. We have:

* Commonly used for topic modelling.
* Note that the number of topics must be pre-specified prior to inference.
* Topics have no automatically assigned names.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/lda.png}
\end{figure}


## Word2Vec (1/4)

Word2vec [@word2vec] is a method to produce word embeddings. Word embeddings 
allow us to project words into a space that has some interesting properties.

* Based on the Skip-gram model proposed by Mikolov in the original paper, which 
  models the probability of a word given the surrounding words (ordering is not 
  important) using a single layer neural network.
* Words having similar meanings are close to each other, and distant from 
  words having very different meanings.
* Word arithmetic is possible. For example one may do the operation 
    
$$
vec("Madrid") - vec("Spain") + vec("France") \sim vec("Paris")
$$


## Word2Vec (2/4)

Word2vec [@word2vec] is a method to produce word embeddings. Word embeddings 
allow us to project words into a space that has some interesting properties.

\begin{figure}
\includegraphics[width=0.9\textwidth]{figures/word2vec_architecture.png}
\end{figure}


## Word2Vec (3/4)

Word2vec [@word2vec] is a method to produce word embeddings. Word embeddings 
allow us to project words into a space that has some interesting properties.

\begin{figure}
\includegraphics[width=0.6\textwidth]{figures/word2vec_space.png}
\end{figure}


## Word2Vec (4/4)

Word2vec [@word2vec] is a method to produce word embeddings. Word embeddings 
allow us to project words into a space that has some interesting properties.

* @pyysalo fit a word2vec model on PubMed, PubMec Central, and biomedical 
  articles in wikipedia and pubblished the resulting biomedical word-embeddings.
* Biomedical embeddings work better on biomedical tasks due to the domain 
  specific content being more more similar to the content in which the 
  algorithms are applied.
* For example, they have led to better performance when classifying sentences 
  as containing ADRs or not [@saldana].


## GloVe (1/2)

Global Vectors for word representation (GloVe) described by @glove that 
use a log-bilinear regression model to model word co-occurrences within a 
context window.

* It was designed to have the attractive properties that enable word arithmetic 
  operations seen in word2vec.
* The authors showed that GloVe can outperform word2vec in the word analogy 
  task.
* Is also easier to parallelize by virtue of its implementation, allowing it 
  to be trained in much larger datasets more easily.
* Like in word2vec, the word vectors obtained with GloVe are fixed and do not 
  change with context.


## GloVe (2/2)

Global Vectors for word representation (GloVe) described by @glove that 
use a log-bilinear regression model to model word co-occurrences within a 
context window.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/glove_vs_word2vec.png}
\end{figure}


## ELMO

ELMO is a model based on pre-training of bi-directional language models (LSTMs) 
to produce context dependent word vectors [@elmo].

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/elmo_nn.png}
\end{figure}


## BERT

BERT [@elmo] is a purely attentional model based on bi-directional transformers 
to produce context dependent word vectors similar to ELMO.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/bert_diagram.png}
\end{figure}


# The State and Outlook of Biomedical NLP


## What can we do well?


## What can we do less well?


## Some Current Topics of Research


# Useful Resources


## Datasets


## Models


# Thank you! Q & A


## References

<div id="refs"></div>


# Backup

## Why Biomedical NLP?

* Most of the information out there is in the form of natural language: 
  scientific papers, clinical notes, social media, textbooks, lectures, 
  websites.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/num_publications_year_cancer.pdf}
\end{figure}


## Why Biomedical NLP?

* Most of the information out there is in the form of natural language: 
  scientific papers, clinical notes, social media, textbooks, lectures, 
  websites.

\begin{figure}
\includegraphics[width=0.7\textwidth]{figures/num_publications_year_combined.pdf}
\end{figure}


## GloVe

Global Vectors for word representation (GloVe) described by @glove that 
use a log-bilinear regression model to model word co-occurrences within a 
context window.

\begin{figure}
\includegraphics[width=0.8\textwidth]{figures/glove_equation.png}
\end{figure}

where $X_ij$ is the number of times word $j$ occurs in the context of word $i$, 
and $b$ are bias terms.


## Agenda

* Biomedical NLP 101: Bags of words (30 mins)
* Deep Learning for Biomedical NLP (30 mins)
* Language Modelling (30 mins)
* Text Classification (30 mins)
* Named Entity Recognition (30 mins)
* Question Answering (30 mins)
* Integrating NLP into survival models (30 mins)
