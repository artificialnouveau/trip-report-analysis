from bertopic import BERTopic
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BertModel, BertTokenizer, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import nltk
# nltk.download('wordnet')

class JournalAnalysis:

    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def clean_text(text):
        # Removing everything except alphabets`
        text = re.sub("[^a-zA-Z]"," ",text)
        
        # Removing whitespaces
        text = ' '.join(text.split())
        
        # Converting text to lowercase
        text = text.lower()
        
        # Removing stop words
        text = ' '.join([word for word in text.split() if word not in (ENGLISH_STOP_WORDS)])
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
        return text

    def topic_modelling(df, text_column):
        # Apply text preprocessing
        df[text_column] = df[text_column].apply(clean_text)
    
        documents = df[text_column].tolist()
    
        # Initializing BERTopic model
        # "english" is the language used for stop words and model
        # n_gram_range is set to (1, 2) to create uni-grams and bi-grams
        # min_topic_size is set to 5, to ignore topics with fewer than 5 documents
        model = BERTopic(language="english", calculate_probabilities=True, n_gram_range=(1, 2), min_topic_size=5)
    
        topics, _ = model.fit_transform(documents)
        return model, topics

    def visualize_topic_distribution(self, topics):
        topic_series = pd.Series(topics)
        topic_counts = topic_series.value_counts()
        plt.figure(figsize=(12, 6))
        topic_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Topic')
        plt.ylabel('Frequency')
        plt.title('Topic Distribution')
        plt.show()


    def visualize_topics_by_words(self, model):
        topic_info = model.get_topic_info()
    
        for index, row in topic_info.iterrows():
            topic_words_probs = model.get_topic(row['Topic'])
    
            words = [word for word, prob in topic_words_probs]
            probs = [prob for word, prob in topic_words_probs]
    
            plt.figure(figsize=(10, 5))
            plt.bar(words, probs, color='lightblue')
            plt.title(f"Topic {row['Topic']}")
            plt.xlabel("Words")
            plt.ylabel("Probability")
            plt.xticks(rotation=90)
            plt.show()


    def visualize_document_projection(self, model, documents, topics):
        embeddings = model._extract_embeddings(documents)
        embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
        plt.figure(figsize=(12, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=topics)
        plt.title('Document Projection by Topic')
        plt.show()

    def visualize_topic_heatmap(self, model):
        similarity_matrix = model.visualize_hierarchy(top_n_topics=20)
        plt.figure(figsize=(12, 8))
        sns.heatmap(similarity_matrix, cmap='YlGnBu')
        plt.title('Topic Heatmap')
        plt.show()

    def find_representative_docs(self, model, documents, topics):
        topic_representatives = model.get_representative_docs()
        for topic in set(topics):
            if topic != -1:
                print(f"Topic {topic}: {documents[topic_representatives[topic]]}")

    def find_representative_topics(self, model, word):
        topic_words = model.get_topic_info()
        representative_topics = topic_words[topic_words['Words'].str.contains(word)]
        print(f"Representative topics for '{word}': {representative_topics['Topic'].values}")

    def create_embeddings(self, data, feature='abstract'):
        model = BERTopic(embedding_model="all-MiniLM-L6-v2", language="english")
        feature_values = data[feature].values.tolist()
        embeddings = model._extract_embeddings(feature_values)
        return embeddings

    def summarize_text_pegasus(self, text):
        model_name = "google/pegasus-xsum"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
        inputs = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
        summary_ids = model.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def summarize_text_chatgpt(self, text):
        prompt = f"Summarize the following text:\n\n{text}"
        model_name = 'gpt-2'  # replace 'gpt-2' with the actual GPT-2 model name you have access to
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=512)
        summary_ids = model.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def summarize_text_bert(self, text):
        summarizer = pipeline('summarization', model='bert-large-uncased')
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
