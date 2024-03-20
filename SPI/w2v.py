from gensim.models import Word2Vec
import pickle

with open("data/w2v/data_w2v4text.pkl","rb") as f:
    documents4text = pickle.load(f)
with open("data/w2v/data_w2v4code.pkl","rb") as f:
    documents4code = pickle.load(f)

model4text = Word2Vec(sentences=documents4text, vector_size=300, window=5, min_count=1, workers=4)
model4code = Word2Vec(sentences=documents4code, vector_size=300, window=5, min_count=1, workers=4)


model4text.save("saved_models/w2v/word2vec4text.model")
model4code.save("saved_models/w2v/word2vec4code.model")