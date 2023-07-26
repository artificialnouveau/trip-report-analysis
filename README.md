# Journal Analytics Library

## Objective

The Journal Analytics Library is a comprehensive solution designed for deep analysis of journal or diary entries. It utilizes advanced machine learning algorithms and natural language processing techniques to decode and understand the underlying meanings, topics, patterns, and emotions conveyed in the entries. The ultimate aim of this library is to transform the complex and sometimes chaotic nature of personal narratives into structured, actionable insights. 

Journal or diary entries are treasure troves of personal experience, individual perception, and intricate emotion. However, navigating through the sheer volume of these entries and deriving useful insights from them can be challenging. This library strives to make this process easier, enabling researchers and curious individuals to unravel the trends, patterns, and sentiments encapsulated in the entries over time.

This library serves as a bridge between narrative data and analytical insights, making it a powerful tool for anyone interested in understanding and analyzing journal or diary entries at scale.

The `JournalAnalysis` class provides a set of functionalities to perform analysis on journal entries, specifically focusing on topic modelling and summarization. The features include:

- Topic modelling using BERTopic
- Visualizing topic distributions
- Visualizing topics by most representative words
- Document projection and clusterization by topic
- Creating a topic heatmap
- Finding most representative documents for each topic
- Finding most representative topics for a specific word
- Creating embeddings for specific features of the dataset
- Text summarization using PEGASUS, GPT-3, and BERT models

## Usage

```python
from journal_analysis import JournalAnalysis

# Initialize the JournalAnalysis class
journal_analysis = JournalAnalysis()

# Perform topic modelling on a DataFrame
model, topics = journal_analysis.topic_modelling(df, 'text_column')

# Visualize topic distribution
journal_analysis.visualize_topic_distribution(topics)

# Visualize topics by most representative words
journal_analysis.visualize_topics_by_words(model)

# Visualize document projection by topic
journal_analysis.visualize_document_projection(model, documents, topics)

# Visualize topic heatmap
journal_analysis.visualize_topic_heatmap(model)

# Find the most representative document for each topic
journal_analysis.find_representative_docs(model, documents, topics)

# Find the most representative topics for a specific word
journal_analysis.find_representative_topics(model, 'word')

# Create embeddings for a specific feature of the dataset
embeddings = journal_analysis.create_embeddings(data, 'abstract')

# Summarize text using PEGASUS
summary_pegasus = journal_analysis.summarize_text_pegasus('text')

# Summarize text using GPT-3
summary_gpt3 = journal_analysis.summarize_text_chatgpt('text')

# Summarize text using BERT
summary_bert = journal_analysis.summarize_text_bert('text')
```

## Dependencies

This project makes use of the following libraries:
- BERTopic
- transformers
- pandas
- seaborn
- matplotlib
- scikit-learn
- torch

---
