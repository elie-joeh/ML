import tensorflow as tf
import numpy as np
from scipy import sparse
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import gensim as gm
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from numpy import savetxt


imdb_train_path = "..\Datasets\IMDB-train.txt"
imdb_validation_path = "..\Datasets\IMDB-valid.txt"
imdb_test_path = "..\Datasets\IMDB-test.txt"
plt.style.use('ggplot')

'''
1) Preparing the data: First of all, transform your dataset to word indices (a word index is simply an integer ID for 
the word). Second, pad the data with 0 so that the size is consistent

2) Create the embedding matrix, using the pre trained vector

3) Create an embedding layer, by setting the weights to the embedding matrix weights, and by setting trainable to false 
so that the weight values don't change. Think about an embedding layer as a loop up table, with word indices as rows
and output dimensions as columns

'''

NUM_WORDS = 10000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 1050

def main():
    imdb_train = fetchData(imdb_train_path)
    imdb_val = fetchData(imdb_validation_path)
    imdb_test = fetchData(imdb_test_path)

    imdb_train = cleanData(imdb_train)
    imdb_val = cleanData(imdb_val)
    imdb_test = cleanData(imdb_test)

    '''
    imdb_train = imdb_train[:5000, :]
    imdb_val = imdb_val[:2000, :]
    imdb_test = imdb_test[:4000, :]
    '''

    X_train, y_train, X_val, y_val, X_test, y_test, word_index = prepareData(imdb_train, imdb_val, imdb_test)

    embedding_matrix, vocabulary_size = generateEmbeddingMatrix(word_index)

    storeEmbeddingMatrix(embedding_matrix)

    '''
    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    model = createModel((embedding_layer))

    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
            epochs=20, batch_size=50,
            shuffle=True,
            verbose=False)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))

    plot_history(history)
    '''

def storeEmbeddingMatrix(embeddingMatrix):
    savetxt('..\Model\embedding_matrix.csv', embeddingMatrix, delimiter=',')



def createModel(embedding_layer):
    model = Sequential()
    #model.add(Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32'))
    model.add(embedding_layer)

    model.add(Conv1D(128, 2, activation='relu'))
    model.add(MaxPooling1D(3))

    model.add(Conv1D(128, 2, activation='relu', input_shape=(MAX_SEQUENCE_LENGTH,)))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['acc'])

    return model

def generateEmbeddingMatrix(word_index):
    word_vec = gm.models.KeyedVectors.load_word2vec_format('../Model/GoogleNews-vectors-negative300.bin', binary=True)
    vocabulary_size = min(len(word_index) + 1, NUM_WORDS)
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= NUM_WORDS:
            continue
        try:
            embedding_vector = word_vec[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

    del word_vec

    return embedding_matrix, vocabulary_size

def plot_history(history):
    print(history.history.items())
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def prepareData(imdb_train, imdb_val, imdb_test):
    tokenizer = Tokenizer(num_words=NUM_WORDS)

    tokenizer.fit_on_texts(imdb_train[:, 0])

    sequences_train = tokenizer.texts_to_sequences(imdb_train[:, 0])
    sequences_val = tokenizer.texts_to_sequences(imdb_val[:, 0])
    sequence_test = tokenizer.texts_to_sequences(imdb_test[:, 0])

    word_index = tokenizer.word_index

    X_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = to_categorical(imdb_train[:, 1])

    X_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
    y_val = to_categorical(imdb_val[:, 1])

    X_test = pad_sequences(sequence_test, maxlen=MAX_SEQUENCE_LENGTH)
    y_test = to_categorical(imdb_test[:, 1])

    return X_train, y_train, X_val, y_val, X_test, y_test, word_index

def cleanData(dataset):
    for row in dataset:
        row[0] = row[0].replace('<br />', ' ').lower()
        row[0] = re.sub(r'\W', ' ', row[0])
        row[0] = re.sub(r'\s+', ' ', row[0])

    return dataset

def fetchData(path):
    dataset = []
    with open(path, mode='r', encoding='utf-8') as data:
        data = data.readlines()
        for line in data:
            line = line.rstrip("\n")
            row = line.split('\t')
            dataset.append(row)
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    return np.array(dataset)

def fetchSparseMatrix(path):
    return np.asarray(sparse.load_npz(path).todense())

if __name__ == "__main__":
    main()