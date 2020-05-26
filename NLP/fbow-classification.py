import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


imdb_fbow_train_path = "..\Datasets\imdb-fbow-train.npz"
imdb_fbow_validation_path = "..\Datasets\imdb-fbow-validation.npz"
imdb_fbow_test_path = "..\Datasets\imdb-fbow-test.npz"


def main():
    imdb_train_new = fetchSparseMatrix(imdb_fbow_train_path)
    imdb_validation_new = fetchSparseMatrix(imdb_fbow_validation_path)
    imdb_test_new = fetchSparseMatrix(imdb_fbow_test_path)

    naiveBayes(imdb_train_new, imdb_validation_new, imdb_test_new)
    #linearsvm(imdb_train_new, imdb_validation_new, imdb_test_new)
    #decistionTree(imdb_train_new, imdb_validation_new, imdb_test_new)
    #logisticRegression(imdb_train_new, imdb_validation_new, imdb_test_new)

'''
Logistic Regression:
0.88636
------------------------
{'alpha': 0.001, 'loss': 'log', 'max_iter': 100, 'n_jobs': 4}

test f1 score: 
f1 score is  0.8905936504189779
'''
def logisticRegression(train_data, validation_data, test_data):
    print("Logistic Regression")
    cv_train_data = np.hstack((train_data.T, validation_data.T)).T

    # this part is commented out since it was used for hyperparameter tuning

    lr_cls = SGDClassifier()
    parameters = {
                'n_jobs': [4],
                'loss': ['log'],
                'alpha': [0.00001, 0.0001, 0.001],
                'max_iter': [100, 1000, 2000, 3000]
                  }
    parameterTuning(lr_cls, parameters, cv_train_data[:, :-1], cv_train_data[:, -1])

    svm_cls = SGDClassifier(n_jobs=4, loss='log', alpha=0.001, max_iter=100)
    svm_cls.fit(cv_train_data[:, :-1], cv_train_data[:, -1])
    testClassifer(svm_cls, test_data[:, :-1], test_data[:, -1])

'''
0.70808
------------------------
{'max_depth': 4000, 'min_samples_leaf': 10, 'min_samples_split': 2}

test f1 score is  0.7047819
'''
def decistionTree(train_data, validation_data, test_data):
    print("Decision Tree")
    cv_train_data = np.hstack((train_data.T, validation_data.T)).T
    '''
    dct_cls = DecisionTreeClassifier()
    
    parameters = {'max_depth': [2000, 4000, 6000],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 5, 10]
                  }

    parameterTuning(dct_cls, parameters, cv_train_data[:, :-1], cv_train_data[:, -1])
    '''
    dct_cls = DecisionTreeClassifier(max_depth=4000, min_samples_split=2, min_samples_leaf=10)
    dct_cls.fit(cv_train_data[:, :-1], cv_train_data[:, -1])
    testClassifer(dct_cls, test_data[:, :-1], test_data[:, -1])


'''
-------------------------
0.87572
------------------------
{'alpha': 0.001, 'max_iter': 2000}

f1 score is  0.8682612467468087
'''
def linearsvm(train_data, validation_data, test_data):
    print("Linear SVM")
    cv_train_data = np.hstack((train_data.T, validation_data.T)).T

    #this part is commented out since it was used for hyperparameter tuning
    ''' 
    svm_cls = SGDClassifier() 
    parameters = {
                  'alpha': [0.00001, 0.0001, 0.001],
                'max_iter' : [50, 100, 1000, 2000, 5000]
                  }
    parameterTuning(svm_cls, parameters, cv_train_data[:, :-1], cv_train_data[:, -1])
    '''
    svm_cls = SGDClassifier(alpha=0.001, max_iter=2000)
    svm_cls.fit(cv_train_data[:, :-1], cv_train_data[:, -1])
    testClassifer(svm_cls, test_data[:, :-1], test_data[:, -1])

'''
{'var_smoothing': 1e-06} --> 0.83352
test with best paramter --> 0.654
'''
def naiveBayes(train_data, validation_data, test_data):
    print("Naive Bayes")
    cv_train_data = np.hstack((train_data.T, validation_data.T)).T
    '''
    gnb_clf = GaussianNB()
    parameters = {
        'var_smoothing' : [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    }

    best_parameters = parameterTuning(gnb_clf, parameters, cv_train_data[:, :-1], cv_train_data[:, -1])
    '''
    gnb_clf = GaussianNB(var_smoothing=1e-6)
    gnb_clf.fit(cv_train_data[:, :-1], cv_train_data[:, -1])
    testClassifer(gnb_clf, test_data[:, :-1], test_data[:, -1])

def parameterTuning(estimator, parameters, train, target):
    gridsearch = GridSearchCV(estimator, parameters)

    gridsearch.fit(train, target)

    print(gridsearch.cv_results_)
    print('----------------------')
    print(gridsearch.best_estimator_)
    print('-------------------------')
    print(gridsearch.best_score_)
    print('------------------------')
    print(gridsearch.best_params_)

    return gridsearch.best_params_

def testClassifer(classifier, test_x_data, test_y_data):
    y_pred = classifier.predict(test_x_data)
    evaluatePrediction(y_pred, test_y_data)
    return

def evaluatePrediction(predictions, y_values):
    correct = 0
    false = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for prediction, y in zip(predictions, y_values):
        if float(y)>0 and prediction == y:
            correct += 1
            true_positive += 1
        elif float(y)>0 and prediction != y:
            false += 1
            false_negative += 1
        elif float(y)<=0 and prediction==y:
            correct += 1
            true_negative += 1
        else:
            false += 1
            false_positive += 1
    accuracy = correct / (correct + false)
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    F1 = (2*precision*recall) / (precision + recall)

    print("Accuracy is: ", accuracy)
    print("Precision is: ", precision)
    print("recall is ", recall)
    print("f1 score is ", F1)

def fetchSparseMatrix(path):
    return np.asarray(sparse.load_npz(path).todense())


if __name__ == "__main__":
    main()