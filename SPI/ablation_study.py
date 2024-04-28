import os
from keras.layers import LSTM, Conv1D, GlobalMaxPooling1D, Dense, Input, Concatenate
from keras.models import Sequential, Model
from keras.metrics import BinaryAccuracy, Precision, Recall, F1Score
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

#============================config==========
EMBEDDING_DIM = 300 
NUM_UNIT = 128
MAX_LEN = 100

#=============================data====
def get_data(path):
    file_list = []
    for root, dirs, fs in os.walk(path):
        for f in fs:
            if f == 'vector_spi.npz':
                file_list.append(os.path.join(root,f))
    
    x_1 = []
    x_2 = []
    y = []
    for f in file_list:
        contents = np.load(f, allow_pickle=True)
        x_1.append(contents['code_vector'])
        x_2.append(contents['message_vector'])
        y.append(contents['label'])
    return x_1, x_2, y


#============================model===================
def create_base_model():
    model = Sequential()
    model.add(LSTM(NUM_UNIT, return_sequences=True, input_shape=(MAX_LEN, EMBEDDING_DIM)))
    #model.add(LSTM(NUM_UNIT, return_sequences=True))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model



#=================================================================

def train(path):

    #===================================Data loading
    x_1, x_2, y = get_data(path)
    x1 = np.array(pad_sequences(x_1, maxlen=200, padding='post', truncating='post'))
    y = np.array(y).reshape(-1,1)

    train_x1, test_x1,\
    train_y, test_y,\
    =train_test_split(x1, y, random_state=2024, test_size=0.2)


    #===================================model loading
    model = create_base_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[BinaryAccuracy, Precision, Recall, F1Score])
    checkpoint = ModelCheckpoint(os.path.join('saved_models/detection4patchdb0421-200token','temp.weights.h5'), monitor='f1_score', verbose=1, save_best_only=True, save_weights_only=True, mode='max',save_freq="epoch")
    model.fit(train_x1, train_y, validation_data=(test_x1, test_y),batch_size=32, epochs=100, validation_split=0.2, callbacks=[checkpoint])
    result = model.evaluate(test_x1, test_y, verbose=False)
    print(result)


def getTP_etcal(labels,predictions):
    tp,fp,tn,fn = 0,0,0,0
    for i,j in zip(labels, predictions):
        if i==1 and j==1:
            tp += 1
        elif i==0 and j==1:
            fp += 1
        elif i == 0 and j==0:
            tn += 1
        else:
            fn += 1
    return tp,fp,tn,fn


def test(path):

    x_1, _, y = get_data(path)
    x1 = np.array(pad_sequences(x_1, maxlen=200, padding='post', truncating='post'))
    y = np.array(y).reshape(-1,1)

    model = create_base_model()
    model.load_weights(os.path.join('saved_models/detection4patchdb0421-200token','temp.weights.h5'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[BinaryAccuracy, Precision, Recall, F1Score])
    eval_results = model.evaluate(x1, y, return_dict=True)
    print(eval_results)
    """
    predictions = model.predict(x1)
    predicitons = (predictions > 0.5).astype(int)
    print(getTP_etcal(y.squeeze(),predicitons.squeeze())) 
    """

def main(mode, train_path, test_path):
    if mode == 1:
        train(train_path)
    else:
        test(test_path)


if __name__ == '__main__':
    mode = 1
    train_path = "/Users/min/Code/SPI/MrSPI/data/1000samples_patchdb/train"
    test_path = "/Users/min/Code/SPI/MrSPI/data/1000samples_patchdb/test"
    main(mode, train_path, test_path)
    
