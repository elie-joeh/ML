from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from scipy import sparse
import re
import numpy as np

imdb_train_path = "..\Datasets\IMDB-train.txt"
imdb_validation_path = "..\Datasets\IMDB-valid.txt"
imdb_test_path = "..\Datasets\IMDB-test.txt"

new_imdb_train_path_fbow = "..\Datasets\IMDB-train-new-fbow.txt"
new_imdb_validation_path_fbow = "..\Datasets\IMDB-valid-new-fbow.txt"
new_imdb_test_path_fbow = "..\Datasets\IMDB-test-new-fbow.txt"

new_imdb_train_path_bbow = "..\Datasets\IMDB-train-new-bbow.txt"
new_imdb_validation_path_bbow = "..\Datasets\IMDB-validation-new-bbow.txt"
new_imdb_test_path_bbow = "..\Datasets\IMDB-test-new-bbow.txt"

imdb_fbow_train_path = "..\Datasets\imdb-fbow-train.npz"
imdb_fbow_validation_path = "..\Datasets\imdb-fbow-validation.npz"
imdb_fbow_test_path = "..\Datasets\imdb-fbow-test.npz"

imdb_bbow_train_path = "..\Datasets\imdb-bbow-train.txt"
imdb_bbow_validation_path = "..\Datasets\imdb-bbow-validation.txt"
imdb_bbow_test_path = "..\Datasets\imdb-bbow-test.txt"

dict_len = 10000

'''
This file creates a pipeline in order to determine the method combination of word representation with the corresponding machine learning method.
Mainly, bag of words, and tfidf are being compared with different ML methods, such as decision trees, logistic regression and SVM.
This is done in order to determine the most performing ML method, and if it uses bow or tfidf, and its best hyperparameters

Then, we vectorize our database depending on best performing parameters.
'''
def main():
    imdb_train = fetchData(imdb_train_path)
    imdb_validation = fetchData(imdb_validation_path)
    imdb_test = fetchData(imdb_test_path)

    imdb_train = cleanData(imdb_train)
    imdb_validation = cleanData(imdb_validation)
    imdb_test = cleanData(imdb_test)

    tuning(imdb_train)
    vectorize(imdb_train, imdb_validation, imdb_test, "imdb")
    return 0

'''
Method that tunes the vectorizer: regular count vectorizer and tfidf are compared with different classifier.
in our case, it's shown that using count vectorizer gives a higher accuracy on all different classifiers. 
'''
def tuning(imdb_train):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier())
    ])

    parameters = {
        "clf__alpha": (5e-6, 1e-05, 5e-5, 4e-5,4e-1),
        "clf__max_iter": (50, 80, 100),
        "vect__max_df": (0.3, 0.5, 0.6),
        "vect__max_features": (None, 5000),
        "vect__ngram_range": [(1, 2)]
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)

    grid_search.fit(imdb_train[:, :-1].ravel(), imdb_train[:, -1])

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def vectorize(train_data, validation_data, test_data, datasetname):
    vectorizer_fbow = CountVectorizer(binary=False, max_features=10000, ngram_range=(1,2), max_df=0.5)
    #vectorizer_bbow = CountVectorizer(binary=True, max_features=10000)


    # fit transform on the train data in order to get our vocab in freq
    fbow_vectors_train = vectorizer_fbow.fit_transform(train_data[:, 0])

    # get the word occurences for both valid and test data sets
    fbow_vectors_validation = vectorizer_fbow.transform(validation_data[:, 0])
    fbow_vectors_test = vectorizer_fbow.transform(test_data[:, 0])

    #bbow_vectors_train = vectorizer_bbow.fit_transform(train_data[:, 0])
    #bbow_vectors_validation = vectorizer_bbow.transform(validation_data[:, 0])
    #bbow_vectors_test = vectorizer_bbow.transform(test_data[:, 0])

    # report the train, test and valid data sets
    reportDatasetVocab(fbow_vectors_train.toarray(), vectorizer_fbow, datasetname)
    #reportDatasetVocab(bbow_vectors_train.toarray(), vectorizer_bbow, "yelp")

    reportDataset(train_data, vectorizer_fbow, new_imdb_train_path_fbow)
    reportDataset(validation_data, vectorizer_fbow, new_imdb_validation_path_fbow)
    #reportDataset(test_data, vectorizer_fbow, new_imdb_test_path_fbow)

    # reportDataset(train_data, vectorizer_bbow, new_imdb_train_path_bbow)
    # reportDataset(validation_data, vectorizer_bbow, new_imdb_validation_path_bbow)
    # reportDataset(test_data, vectorizer_bbow, new_imdb_test_path_bbow)

    FBOWRepresentation(fbow_vectors_train, train_data[:,-1], fbow_vectors_validation, validation_data[:, -1], fbow_vectors_test, test_data[:, -1], datasetname)
    #BBOWRepresentation(bbow_vectors_train, train_data[:, -1], bbow_vectors_validation, validation_data[:, -1], bbow_vectors_test, test_data[:, -1], datasetname)


def BBOWRepresentation(bbow_vectors_train, train_labels, bbow_vectors_validation, validation_labels, bbow_vectors_test, test_labels, datasetname):
    bbow_vectors_train = bbow_vectors_train.toarray()
    bbow_vectors_validation = bbow_vectors_validation.toarray()
    bbow_vectors_test = bbow_vectors_test.toarray()

    bbow_vectors_train = np.vstack((bbow_vectors_train.T.astype('int'), train_labels.astype('int'))).T
    bbow_vectors_validation = np.vstack((bbow_vectors_validation.T.astype('int'), validation_labels.astype('int'))).T
    bbow_vectors_test = np.vstack((bbow_vectors_test.T.astype('int'), test_labels.astype('int'))).T

    writeBOWdata(bbow_vectors_train, imdb_bbow_train_path)
    writeBOWdata(bbow_vectors_validation, imdb_bbow_validation_path)
    writeBOWdata(bbow_vectors_test, imdb_bbow_test_path)


#function which will generate the FBOW representation of the yelp data sets
def FBOWRepresentation(fbow_vectors_train, train_labels, fbow_vectors_validation, validation_labels, fbow_vectors_test, test_labels, dataset):
    print('fbow presentation')
    fbow_vectors_train = fbow_vectors_train.toarray()
    fbow_vectors_validation = fbow_vectors_validation.toarray()
    fbow_vectors_test = fbow_vectors_test.toarray()

    fbow_vectors_train = np.vstack((fbow_vectors_train.T, train_labels.astype('float64'))).T
    fbow_vectors_validation = np.vstack((fbow_vectors_validation.T, validation_labels.astype('float64'))).T
    fbow_vectors_test = np.vstack((fbow_vectors_test.T, test_labels.astype('float64'))).T

    fbow_vectors_train_sparse = sparse.csc_matrix(fbow_vectors_train)
    fbow_vectors_validation_sparse = sparse.csc_matrix(fbow_vectors_validation)
    fbow_vectors_test_sparse = sparse.csc_matrix(fbow_vectors_test)

    sparse.save_npz(imdb_fbow_train_path, fbow_vectors_train_sparse)
    sparse.save_npz(imdb_fbow_validation_path, fbow_vectors_validation_sparse)
    sparse.save_npz(imdb_fbow_test_path, fbow_vectors_test_sparse)


def writeBOWdata(data, path):
    file = open(path, mode='w')
    for review in data:
        review_txt = review[:-1]
        review_label = review[-1]
        review_str = ' '.join(str(x) for x in review_txt)
        file.write(review_str + "\t" + str(review_label) + "\n")
    file.close()


def reportDataset(dataset, vectorizer, path):
    vocab = vectorizer.get_feature_names()

    file = open(path, mode='w')
    for review in dataset:
        new_review = []
        review_text = review[0].split(' ')
        review_label = review[1]

        for word in review_text:
            if word in vocab:
                new_review.append(vectorizer.vocabulary_.get(word))

        review_str = ' '.join(str(x) for x in new_review)
        file.write(review_str + "\t" + str(review_label) + "\n")

    file.close()


def reportDatasetVocab(train_data, vectorizer, datasetname):
    # get the vocab
    vocab = vectorizer.get_feature_names()
    # sum the vector by column to get the word freq
    word_frequencies = train_data.sum(axis=0)

    if datasetname == "imdb":
        file = open("..\Datasets\IMDB-vocab.txt", mode="w")
    else:
        file = open("..\Datasets\YELP-vocab.txt", mode="w")

    for index in range(len(vocab)):
        word = vocab[index]
        word_freq = word_frequencies[index]
        word_ID = vectorizer.vocabulary_.get(word)
        file.write(str(word) + "\t" + str(word_ID) + "\t" + str(word_freq) + "\n")

    file.close()

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

    return np.array(dataset)


if __name__ == "__main__":
    main()