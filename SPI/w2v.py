from gensim.models import Word2Vec

with open("data_w2v4text.txt") as f:
    documents4text = f.readlines()
with open("data_w2v4code.txt") as f:
    documents4code = f.readlines()

model4text = Word2Vec(sentences=documents4text, vector_size=300, window=5, min_count=1, workers=4)
model4code = Word2Vec(sentences=documents4code, vector_size=300, window=5, min_count=1, workers=4)


model4text.save("word2vec4text.model")
model4code.save("word2vec4code.model")