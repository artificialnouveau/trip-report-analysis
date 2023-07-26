from journal_analysis_function import JournalAnalysis

# Initializing the class
api_key = 'YOUR_API_KEY'
journal_analysis = JournalAnalysis(api_key)

# Text data
text_data = "This is a long piece of text that you want to summarize. It could be an article, a research paper, or a book."

# DataFrame (you need to have a real DataFrame here)
df = pd.DataFrame({"text": [text_data, text_data, text_data]})

# Using the topic modelling method
model, topics = journal_analysis.topic_modelling(df, 'text')

# Visualize the topic distribution
journal_analysis.visualize_topic_distribution(topics)

# Visualize topics by words
journal_analysis.visualize_topics_by_words(model)

# Visualize document projection
journal_analysis.visualize_document_projection(model, df['text'].tolist(), topics)

# Visualize topic heatmap
journal_analysis.visualize_topic_heatmap(model)

# Find representative documents
journal_analysis.find_representative_docs(model, df['text'].tolist(), topics)

# Find representative topics for a word
journal_analysis.find_representative_topics(model, 'text')

# Create embeddings
embeddings = journal_analysis.create_embeddings(df, 'text')

# Text summarization using PEGASUS
summary_pegasus = journal_analysis.summarize_text_pegasus(text_data)
print(summary_pegasus)

# Text summarization using GPT-2 (Replace 'gpt-2' with the GPT-2 model you have access to)
#summary_gpt = journal_analysis.summarize_text_chatgpt(text_data)
#print(summary_gpt)

# Text summarization using BERT
summary_bert = journal_analysis.summarize_text_bert(text_data)
print(summary_bert)
