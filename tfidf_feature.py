import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# file reading and preprocessing
x = pickle.load(open("essays_mairesse.p", "rb"))
revs, W, W2, word_idx_map, vocab, mairesse = x[0], x[1], x[2], x[3], x[4], x[5]
for item in revs:
    delete_list = []
    for st in item['text']:
        if len(st)<10:
            delete_list.append(st)
    for i in delete_list:
        item['text'].remove(i)

# vecorization
text = [". ".join(item['text']) for item in revs]
vectorizer_doc = TfidfVectorizer(ngram_range=(1, 3), lowercase=True, stop_words="english", max_features=200)
vectorizer_doc.fit(text)
X = vectorizer_doc.transform(text)
tfidf = {}
for i in range(len(revs)):
    tfidf[revs[i]['user']] = X[i].toarray().reshape(-1)

# saving data    
with open('tfidf.p', 'wb') as handle:
    pickle.dump(tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)
