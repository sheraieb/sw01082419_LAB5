from collections import Counter #count word occurrences
import pandas as pd

file_path = "Processed_Reviews.csv"
df = pd.read_csv(file_path)

tokenized_reviews = df['tokenized'].dropna().apply(eval)

all_words = [word for review in tokenized_reviews for word in review]
unique_words = list(set(all_words))

word_freq = Counter(all_words)
sorted_word_freq = dict(sorted(word_freq.items(), key=lambda item: item[1],
reverse=True))

document_vectors = []
for review in tokenized_reviews:
 document_vector = [1 if word in review else 0 for word in sorted_word_freq.keys()]
 document_vectors.append(document_vector)

doc_vectors_df = pd.DataFrame(document_vectors, columns=sorted_word_freq.keys())

doc_vectors_df.to_csv("document_vectors.csv", index=False)


word_freq_df = pd.DataFrame(list(sorted_word_freq.items()), columns=["Word",
"Frequency"])
print("Word Frequency Table:")
print(word_freq_df)



