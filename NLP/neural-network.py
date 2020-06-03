import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy import sparse
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler

imdb_fbow_train_path = "./datasets/imdb-fbow-train.npz"
imdb_fbow_validation_path = "./datasets/imdb-fbow-validation.npz"
imdb_fbow_test_path = "./datasets/imdb-fbow-test.npz"

plt.style.use('ggplot')

input_dim = 0

def main():
    global input_dim
    imdb_train_new = fetchSparseMatrix(imdb_fbow_train_path)
    imdb_validation_new = fetchSparseMatrix(imdb_fbow_validation_path)
    imdb_test_new = fetchSparseMatrix(imdb_fbow_test_path)

    np.random.shuffle(imdb_train_new)
    np.random.shuffle(imdb_validation_new)
    np.random.shuffle(imdb_test_new)

    input_dim = imdb_train_new[:, :-1].shape[1]

    #hyperparameters to tune
    #number_nodes = [[100, 100, 10, 1], [200, 100, 10, 1], [100, 100, 100, 1], [300, 100, 50, 1]]
    optimizer = [keras.optimizers.Adam(amsgrad=True)]
    weight_decay = [1e-7, 1e-6, 1e-5]
    epochs = [5, 10, 20]
    batch_size = [500, 1000]
    dropout_rate = [0.8]
    #optimizer = [keras.optimizers.Adam(learning_rate=0.1), keras.optimizers.Adam(learning_rate=0.01)]

    model = KerasClassifier(build_fn=create_model, verbose=0)

    callbacks_list = [LearningRateScheduler(lr_schedule)]

    param_grid = dict(weight_decay=weight_decay, epochs=epochs, batch_size=batch_size, dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid.fit(imdb_train_new[:, :-1], imdb_train_new[:, -1])

    #summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    '''
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
    '''

def create_model(weight_decay=0.1, dropout_rate=0.7):
    model = Sequential()
    model.add(layers.Dense(200, input_dim=input_dim, activity_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    '''
    model.add(layers.Dense(50, activity_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))
    '''

    model.add(layers.Dense(150, activity_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1, activity_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()

    return model


def lr_schedule(epoch):
    lrate = 0.1
    if epoch > 5:
        lrate = 0.01
    if epoch > 15:
        lrate = 0.001
    return lrate

'''
def exp_decay(epoch):
  lrate = 0.001
  k = 0.1
  lrate = lrate * exp(-k*t)
  return lrate
'''

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