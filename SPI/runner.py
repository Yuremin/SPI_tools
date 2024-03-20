from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Activation
import numpy as np


word2vec4text_model = Word2Vec.load("saved_models/w2v/word2vec4text.model")
word2vec4code_model = Word2Vec.load("saved_models/w2v/word2vec4code.model")

EMBEDDING_DIM = word2vec4text_model.vector_size  # 嵌入向量的维度
NUM_UNIT = 128
MAX_LEN = 100

def get_model():
    model = Sequential()
    model.add(Embedding(EMBEDDING_DIM, NUM_UNIT, input_length=MAX_LEN))
    model.add(LSTM(NUM_UNIT), return_sequences=True))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Make predictions
predictions = model.predict(X_train[:5])

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
