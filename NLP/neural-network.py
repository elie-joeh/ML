import tensorflow as tf
import numpy as np
from scipy import sparse
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

imdb_fbow_train_path = "..\Datasets\imdb-fbow-train.npz"
imdb_fbow_validation_path = "..\Datasets\imdb-fbow-validation.npz"
imdb_fbow_test_path = "..\Datasets\imdb-fbow-test.npz"

plt.style.use('ggplot')

def main():
    imdb_train_new = fetchSparseMatrix(imdb_fbow_train_path)
    imdb_validation_new = fetchSparseMatrix(imdb_fbow_validation_path)
    imdb_test_new = fetchSparseMatrix(imdb_fbow_test_path)

    np.random.shuffle(imdb_train_new)
    np.random.shuffle(imdb_validation_new)
    np.random.shuffle(imdb_test_new)

    model = create_model(imdb_train_new[:, :-1].shape[1])

    history = model.fit(imdb_train_new[:, :-1],
                        imdb_train_new[:, -1],
                        epochs=30,
                        shuffle=True,
                        verbose=False,
                        validation_data=(imdb_validation_new[:, :-1], imdb_validation_new[:, -1]),
                        batch_size=10)

    loss, accuracy = model.evaluate(imdb_validation_new[:, :-1], imdb_validation_new[:, -1], verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(imdb_test_new[:, :-1], imdb_test_new[:, -1], verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    plot_history(history)


def create_model(input_dim):
    model = Sequential()
    model.add(layers.Dense(100, input_dim=input_dim, activity_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(10, activity_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(1, activity_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
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


def fetchSparseMatrix(path):
    return np.asarray(sparse.load_npz(path).todense())

if __name__ == "__main__":
    main()