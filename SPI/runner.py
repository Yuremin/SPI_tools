from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Activation
import numpy as np

# 假设您已经加载了Word2Vec模型
word2vec_model = Word2Vec.load("path_to_your_word2vec_model.model")

# 获取词汇大小和嵌入维度
VOCAB_SIZE = len(word2vec_model.wv)  # 词汇大小
EMBEDDING_DIM = word2vec_model.vector_size  # 嵌入向量的维度

# 初始化嵌入矩阵
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
for i, word in enumerate(word2vec_model.wv.index_to_key):
    embedding_vector = word2vec_model.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# 模型构建
model = Sequential()
# 添加嵌入层，weights参数设置为预训练的嵌入权重
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=100, weights=[embedding_matrix], trainable=False))

# LSTM层
model.add(LSTM(128, return_sequences=True))

# CNN层
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())

# Softmax输出层
model.add(Dense(2, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型摘要
model.summary()


import numpy as np

# Generate synthetic data
np.random.seed(42)  # For reproducibility

# Assuming we're working with a small dataset of 1000 examples
num_samples = 1000

# Generate random sequences of integers to simulate tokenized text data
X_train = np.random.randint(1, VOCAB_SIZE, (num_samples, MAX_LENGTH))

# Generate binary labels for our synthetic dataset
y_train = np.random.randint(2, size=(num_samples, 1))

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)



# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Make predictions
predictions = model.predict(X_train[:5])

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)
