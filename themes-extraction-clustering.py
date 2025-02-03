#!/usr/bin/env python
# coding: utf-8

# # Themes extraction with pke
# 
# **Themes** are topical keywords and phrases that are prominent in a document.
# 
# To extract the themes we will use the [pke - Python keyphrase extraction](https://github.com/boudinfl/pke) toolkit. pke requires [SpaCy](https://spacy.io/usage) and a SpaCy model for the language of the document.
# 
# Let's install spacy and pke first.

# In[1]:


import sys

get_ipython().system('{sys.executable} -m pip install spacy')
get_ipython().system('{sys.executable} -m spacy download en_core_web_sm  # download the English SpaCy model')
get_ipython().system('{sys.executable} -m pip install git+https://github.com/boudinfl/pke.git')


# If you plan to use pke on a command-line installation of Python, you can use the following commands instead:
# 
# ```
# pip install spacy
# python -m spacy download en_core_web_sm
# pip install git+https://github.com/boudinfl/pke.git
# ```

# Let's see how pke works. For this, we are going to use a raw text file called [wiki_gershwin.txt](wiki_gershwin.txt). We first import the module and initialize the keyphrase extraction model (here: TopicRank):

# In[61]:


import pke

extractor = pke.unsupervised.TopicRank()


# Load the content of the document, here document is expected to be in raw format (i.e. a simple text file). The document is automatically preprocessed and analyzed with SpaCy, using the language given in the parameter:

# In[62]:


text = "The latest smartphone by Apple offers cutting-edge features and a sleek design."
extractor.load_document(text, language='en')


# The keyphrase extraction consists of three steps:
# 
# 1. Candidate selection:  
# With TopicRank, the default candidates are sequences of nouns and adjectives (i.e. `(Noun|Adj)*`)
# 
# 2. Candidate weighting:  
# With TopicRank, this is done using a random walk algorithm.
# 
# 3. N-best candidate selection:  
# The 10 highest-scored candidates are selected. They are returned as (keyphrase, score) tuples.

# In[64]:


extractor.candidate_selection()
extractor.candidate_weighting()
keyphrases = extractor.get_n_best(n=10)

print("Extracted themes:")
print("=================")
for keyphrase in keyphrases:
    print(f'{keyphrase[1]:.5f}   {keyphrase[0]}')


# Next, you can try out different methods for extracting themes: supervised, unsupervised, graph. Compare the themes extracted. If your texts are in other languages than English, test the themes extraction for them and assess the quality. Is this something you might want to use for your final project?
# 
# You can read more about the pke toolkit from their paper ([Boudin, 2016](https://aclanthology.org/C16-2015.pdf)).

# In[2]:


# JUPYTER NOTEBOOK CELL 1

# ===========================
# 1. IMPORTING THE LIBRARIES
# ===========================

import os
import pke
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# For hierarchical clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns

# Set up display defaults
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_theme(style="whitegrid")


# In[ ]:





# In[57]:


documents = [
    # Document 1: About a new smartphone (Technology; Serious)
    "The latest smartphone by Apple offers cutting-edge features and a sleek design.",
    
    # Document 2: About a surprising sports upset (Sports; Exciting/Dramatic)
    "In a surprising upset, the underdog football team clinched the championship title.",
    
    # Document 3: About a culinary event (Food; Optimistic/Joyful)
    "A delicious fusion of flavors in today's culinary event left food critics delighted.",
    
    # Document 4: About tech startups (Technology; Optimistic/Joyful)
    "Tech startups are revolutionizing the industry with innovative software solutions.",
    
    # Document 5: About an intense basketball match (Sports; Exciting/Dramatic)
    "The intense basketball match kept fans on the edge of their seats until the final buzzer."
]



labels =[0,1,2,0,1]


# In[ ]:





# In[50]:


# JUPYTER NOTEBOOK CELL 3

# ===========================================
# 3. EXTRACTING TOPICS USING pke.TopicRank()
# ===========================================

# We define a function that extracts the top N keyphrases (topics)
# from a given text using TopicRank. We’ll store them along with
# their weights.

def extract_topics(text, n_keyphrases=10, language='en'):
    extractor = pke.unsupervised.TopicRank()
    # Load the document into the extractor
    extractor.load_document(input=text, language=language)
    # Select candidates (words or phrases)
    extractor.candidate_selection()
    # Weight the candidates
    extractor.candidate_weighting()
    # Get the top n_keyphrases
    keyphrases = extractor.get_n_best(n=n_keyphrases)
    return keyphrases

# Let's store the extracted topics for each document
all_docs_topics = []
for i, doc_text in enumerate(documents):
    keyphrases = extract_topics(doc_text, n_keyphrases=10, language='en')
    all_docs_topics.append(keyphrases)
    # Print a summary for each document
    print(f"Document {i+1} - Extracted topics:")
    print("==================================")
    for kp in keyphrases:
        print(f"{kp[1]:.5f}   {kp[0]}")
    print("\n")


# In[51]:


# Suppose we already have docs_topics: a list where each element is a list of (topic, weight)
# e.g., docs_topics[i] = [("machine learning", 0.25), ("data", 0.20), ...]

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans

# 1. Load a sentence transformer model (or any other embedding model)
model = SentenceTransformer('all-MiniLM-L6-v2')


# In[52]:


document_embeddings = []

for topics in all_docs_topics:
    print(topics)
    # topics is a list of (topic_string, weight)
    phrases = [t[0] for t in topics]  # just the phrase texts
    # 2. Get embeddings for each phrase
    phrase_embeddings = model.encode(phrases)
    
    # 3. Optionally weight embeddings by score
    weights = np.array([t[1] for t in topics]).reshape(-1, 1)  # shape (n_topics, 1)
    weighted_embeddings = phrase_embeddings * weights
    
    # 4. Aggregate to get a single document vector
    doc_embedding = np.mean(weighted_embeddings, axis=0)  # or sum, or max, etc.
    
    # 5. Store result
    document_embeddings.append(doc_embedding)


# In[ ]:





# In[53]:


# JUPYTER NOTEBOOK CELL 5

# ==============================
# 5. COMPUTE SIMILARITY MEASURES
# ==============================

# Use the scaled data to compute similarities
similarity_matrix = cosine_similarity(document_embeddings)

# Convert to dataframe for easy display
df_similarity = pd.DataFrame(similarity_matrix)

print("Cosine Similarity Matrix:")
df_similarity


# In[59]:


# JUPYTER NOTEBOOK CELL 6

# =================
# 6. K-MEANS CLUSTER
# =================

# We will try K=2 or K=3 (as an example). 
# In practice, you can use domain knowledge or statistical criteria to choose K.

k = 3  # try 2 clusters
kmeans = KMeans(n_clusters=k, random_state=42)
labels_kmeans = kmeans.fit_predict(df_similarity)

# Visualize the cluster assignments
df_clusters_kmeans = pd.DataFrame({'Document': df_similarity.index, 
                                   'Cluster': labels_kmeans, 
                                  'Label': labels})
df_clusters_kmeans


# In[55]:


# JUPYTER NOTEBOOK CELL 7

# ==========================================================
# 7. HIERARCHICAL CLUSTERING (Single, Complete, Average, Ward)
# ==========================================================

# We’ll apply different linkage criteria for hierarchical clustering:
# - 'single'
# - 'complete'
# - 'average'
# - 'ward'

methods = ['single', 'complete', 'average', 'ward']

# We'll store the linkages in a dictionary to examine or plot them
linkage_results = {}
for method in methods:
    Z = linkage(df_similarity, method=method)
    linkage_results[method] = Z

# Let's visualize the dendrogram for each method in a small subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, method in zip(axes, methods):
    dendrogram(linkage_results[method], 
               labels=df_similarity.index.to_list(),
               ax=ax,
               leaf_rotation=90)
    ax.set_title(f"Hierarchical Clustering - {method}")

plt.tight_layout()
plt.show()


# In[58]:


# JUPYTER NOTEBOOK CELL 8

# ============================================
# 8. CUTTING THE DENDROGRAM AND CLUSTER LABELS
# ============================================

# Suppose we want to automatically cut the dendrogram into 2 clusters 
# (you can choose any number you want). We can extract these labels.

num_clusters = 3
for method_of_choice in ['single','complete','average','ward']:
    Z_chosen = linkage_results[method_of_choice]
    
    labels_hier = fcluster(Z_chosen, t=num_clusters, criterion='maxclust')
    
    df_clusters_hier = pd.DataFrame({'Document': df_similarity.index, 
                                     'Cluster': labels_hier, 
                                     'Label': labels})
    print(method_of_choice, df_clusters_hier)


# ## 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




