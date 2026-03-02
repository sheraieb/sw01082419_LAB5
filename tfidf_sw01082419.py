#-------------------------------------what i have learned from this lab's resulted files----------------------#

# first from the tf_scores.csv i learned how frequently each word appears inside of the documents individually, this helped in determining what words dominated every document

# second from the idf_scores.csv i learned how rare a word is across all documents. the score is HIGHER for words that appear in fewer documents

# thirdly from the tfidf_scores.csv i learned how the TF and IDF combine. Its score highlights words that are frequent in ONE document yet RARE across the combined dataset

# lastly i realised that tf_scores and idf scores would be the same in the case of a very small dataset and it is hard for them to reach the same number



import pandas as pd
import math # use for log function
from collections import Counter


file_path = "Processed_Reviews.csv"
df = pd.read_csv(file_path)

tokenized_reviews = df['tokenized'].dropna().apply(eval)

def compute_tf(document):
 word_count = Counter(document)
 tf = {word: count / len(document) for word, count in word_count.items()}
 return tf

def compute_idf(documents):
 N = len(documents) # Total number of documents
 idf = {}
 all_words = set(word for doc in documents for word in doc) # Unique words
 for word in all_words:
  count = sum(1 for doc in documents if word in doc)
  idf[word] = math.log(N / count)
 return idf

def compute_tfidf(document, idf):
 tfidf = {}
 tf = compute_tf(document) # Get TF values for the document
 for word, tf_value in tf.items():
  tfidf[word] = tf_value * idf[word] # Multiply TF and IDF
 return tfidf

documents = tokenized_reviews.tolist()


tf_data = [compute_tf(doc) for doc in documents]
tf_df = pd.DataFrame(tf_data).fillna(0)
tf_df.to_csv("tf_scores.csv", index=False)


idf = compute_idf(documents)
idf_df = pd.DataFrame([idf]).fillna(0)
idf_df.to_csv("idf_scores.csv", index=False)


tfidf_data = [compute_tfidf(doc, idf) for doc in documents]
tfidf_df = pd.DataFrame(tfidf_data).fillna(0)
tfidf_df.to_csv("tfidf_scores.csv", index=False)


