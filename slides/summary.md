% PyPharma NLP Tutorial 2019: \
  Learning by doing NLP
% Diego Saldana, PHC Analytics
% November 2019


# Agenda

- PyPharma Conference 2019
- Natural Language Processing
- PyPharma NLP 2019
- Notebooks Format
- Examples
    - Masked Language Modelling
    - Detecting Adverse Events in the Literature
    - Extracting Disease Mentions
    - Extracting Gene-Disease Associations
    - Question Answering


# PyPharma Conference 2019 (1/2)

- Aimed at **python** users in the pharmaceutical **industry** and 
  **academia**, all aspects of the pharmaceutical **lifecycle** and data 
  **modalities**.
- Attendance is estimated at around **100** people (still growing).
- **Attendants** and **speakers** from Roche, Novartis, GSK, AstraZeneca, 
  UNIBAS, UNIL, IBM, ETHZ, UZH, SIB, and various other companies and 
  universities.


# PyPharma Conference 2019 (2/2)

- Two days, **single** track, **invite** only, **free** to attend.
- **Hosts:** Roche, Novartis, and the University of Basel.
- Day 1: November 21st at the **University of Basel**, Day 2: November 22nd at 
  the **Roche** Viaduktstrasse amphitheater.


# Natural Language Processing (1/2)

- **Automate** the processing of data in **natural language** form.
- Look at **text** as a **database** that we want to be able to **query** in 
  various ways.
- **Combines** elements of linguistics, computer science, statistics, 
  artificial intelligence, etc.
- One natural way to do this is to use **machine learning** to perform 
  **tasks** such as text classification, named entity extraction, relation 
  extraction, etc.


# Natural Language Processing (2/2)

- In recent years, **deep learning** based methods have shown great promise in 
  terms of performance.
- But also the ability to **transfer knowledge** across datasets as well as 
  across tasks.
- Examples include models such as 
    - word2vec
    - GloVe
    - ELMO
    - BERT
    - etc


# Natural Language Processing (3/2)

- The usual **transfer learning** procedure is
    - Step 1: train a **base model** to perform **generic** tasks on very 
      **large** datasets (websites, books, wikipedia, social media, etc).
    - Step 2: **fine tune** the model to perform a **specific** task on a 
      **smaller** dataset.
- **Extensions** to these methods to **biomedical** applications often follow 
  (e.g. Pyysalo embeddings for word2vec, BioBERT for BERT, etc).


# PyPharma NLP 2019

- A **tutorial** on Pharma and Biomedical NLP that will take place on Day 1 of 
  PyPharma (November 21st at the University of Basel).
- The goal is for intermediate pharmaceutical and biomedical python users to 
  **learn** how to do **state of the art** NLP by **doing** it.
- Since doing it is usually **hard**, we provide **tools** for them to make the 
  process **easier**.
- We provide **notebooks** with examples of various **tasks** and **datasets**.


# Notebooks: Format (1/2)

- They will be hosted in **Google Colab** as well as Azure Notebooks.
- **No need** to install anything in your computer, or buy a new one, to do 
  **state of the art**, deep learning based NLP (yay!).
- Prior to using the notebooks, an **introduction** to deep learning based NLP 
  will be done.


# Notebooks: Format (2/2)

- The common **structure** will be:
    - **Downloading** the training datasets.
    - Exploring the **training data** and how it is seen by the model 
      (**inputs** and **outputs**).
    - Training the model and **storing** the results in a **checkpoint**.
    - Re-loading the checkpoint and performing **predictions** on new data, as 
      well as **exploring** the results **interactively**.


# Examples: Masked Language Modelling

<!--- Predicting a missing word by looking at its context.
- Generic task performed to train a base model on large datasets.
- May look useless at first glance but allows the model to learn language 
  structure first. -->

![](figures/summary/language_modelling.png)


# Examples: Detecting Adverse Events

![](figures/summary/text_classification.png)


# Examples: Extracting Disease Mentions

![](figures/summary/named_entity_recognition.png)


# Examples: Extracting Gene-Disease Associations

![](figures/summary/relation_extraction.png)


# Examples: Question Answering (1/2)

![](figures/summary/question_answering_text.png)


# Examples: Question Answering (2/2)

![](figures/summary/question_answering_qa.png)
