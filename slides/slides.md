% PyPharma NLP Workshop 2019
% Diego Saldana
% November 2019


# Introduction to Biomedical NLP


## Why Biomedical NLP? (1/3)

- Most of the information out there is in the form of natural language: 
  scientific papers, clinical notes, social media, textbooks, lectures, 
  websites.

\begin{figure}
    \includegraphics[width=0.7\textwidth]{figures/num_publications_year.pdf}
\end{figure}


## Why Biomedical NLP? (2/3)

- Most of the information out there is in the form of natural language: 
  scientific papers, clinical notes, social media, textbooks, lectures, 
  websites.

\begin{figure}
    \includegraphics[height=0.5\textheight]{figures/wikipedia_timeline.png}
\end{figure}


## Why Biomedical NLP? (3/3)

- Most of the information out there is in the form of natural language: 
  scientific papers, clinical notes, social media, textbooks, lectures, 
  websites.
- This information is potentially very useful but cannot readily be used 
  programmatically and stored in databases, searched, or analyzed.
- As a result this valuable information is "locked into a vault" until a human 
  reads it, structures it and puts it into some database.
- And even when that happens, the scope in which the data can be used is 
  usually limited and chosen by the extractors.
- How can machines help?


## Humans vs. Machines (1/2)

- Machines and humans have different strengths and weaknesses when processing 
  text.

\begin{figure}
    \includegraphics[height=0.7\textheight]{figures/ade_corpus_iaa.png}
\end{figure}


## Humans vs. Machines (2/2)

- Machines and humans have different strengths and weaknesses when processing 
  text.
- Machines in particular are capable of processing vast amounts of text in a 
  very short period of time in a very consistent way and performing simple 
  tasks.
- Humans are take much more time to process text and are less consistent, 
  however they are capable of much more complex reasoning and understanding.


## Humans vs. Machines (3/2)

What are some examples of tasks can computers perform well in 2019?

- Categorizing documents (e.g. automatically assigning MeSH headings to 
  PubMed abstracts)
- Extracting entities from text (e.g. extracting Drugs, Diseases from 
  PubMed abstracts)
- Extracting relations from text (e.g. extracting Adverse Events from 
  PubMed abstracts)
- Answering simple questions based on a small amount of context (e.g. 
  "Which drug should be used as an antidote in benzodiazepine overdose?")


## Some Common Tasks

Language Modelling: A language model assigns probabilities to sequences of 
tokens, where tokens $t$ can be words, characters, sub-words, etc: 

$P\Big(t_{1}, t_{2}, t_{3}, ..., t_{N}\Big)$.

One common way to do this is to decompose this as the probability of the next 
token in the sequence $t_{i}$ given the probability of the sequence up to the 
previous token and some parameters $\Theta$ for our model:

$P\Big(t_{i} | t_{1}, t_{2}, t_{3}, ..., t_{i - 1}, \Theta\Big).$


## References

WIP.


# Backup

## Why Biomedical NLP?

- Most of the information out there is in the form of natural language: 
  scientific papers, clinical notes, social media, textbooks, lectures, 
  websites.

\begin{figure}
    \includegraphics[width=0.7\textwidth]{figures/num_publications_year_cancer.pdf}
\end{figure}


## Why Biomedical NLP?

- Most of the information out there is in the form of natural language: 
  scientific papers, clinical notes, social media, textbooks, lectures, 
  websites.

\begin{figure}
    \includegraphics[width=0.7\textwidth]{figures/num_publications_year_combined.pdf}
\end{figure}


## Agenda

* Biomedical NLP 101: Bags of words (30 mins)
* Deep Learning for Biomedical NLP (30 mins)
    * Language Modelling (30 mins)
    * Text Classification (30 mins)
    * Named Entity Recognition (30 mins)
    * Question Answering (30 mins)
    * Integrating NLP into survival models (30 mins)
