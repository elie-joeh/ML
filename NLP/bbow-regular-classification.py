import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

imdb_bbow_train_path = "..\Datasets\imdb-bbow-train.txt"
imdb_bbow_validation_path = "..\Datasets\imdb-bbow-validation.txt"
imdb_bbow_test_path = "..\Datasets\imdb-bbow-test.txt"

'''
This file learns to classify imdb reviews, by using the vectorized dataset. This is done to evaluate a basic binary BOW
'''
def main():

    imdb_train_new = fetchData(imdb_bbow_train_path)
    imdb_validation_new = fetchData(imdb_bbow_validation_path)
    imdb_test_new = fetchData(imdb_bbow_test_path)
    #randomClassifier(imdb_train_new[:, 0], imdb_train_new[:, 1], imdb_validation_new[:, 0], imdb_validation_new[:, 1])
    #decistionTree(imdb_train_new, imdb_validation_new, imdb_test_new)
    linearsvm(imdb_train_new, imdb_validation_new, imdb_test_new)
    #naiveBayes(imdb_train_new, imdb_validation_new, imdb_test_new)

    return 0


def testClassifer(classifier, test_x_data, test_y_data):
    y_pred = classifier.predict(test_x_data)
    evaluatePrediction(y_pred, test_y_data)
    return


#best {'var_smoothing': 5e-07}
#test f1 = 0.684 vs cross validation F1: 0.76344
def naiveBayes(train_data, validation_data, test_data):
    print("Naive Bayes")
    cv_train_data = np.hstack((train_data.T, validation_data.T)).T.astype('int')
    gnb_clf = GaussianNB()

    parameters = {
        'var_smoothing' : [5e-7]
    }
    best_parameters = parameterTuning(gnb_clf, parameters, cv_train_data[:, :-1], cv_train_data[:, -1])

    gnb_clf = GaussianNB(var_smoothing=5e-7)
    gnb_clf.fit(cv_train_data[:, :-1], cv_train_data[:, -1])
    testClassifer(gnb_clf, test_data[:, :-1].astype('int'), test_data[:, -1].astype('int'))


'''
C parameter: tells the algorithm how much you care about misclassified points. if C is infinite, we get a hard margin instead of soft. which means won't converge
if classes overlap!
gamma: kernel coefficient. if it's high, the class boundaries would be very tight (so any small variation of a new point can end up in another class).

best params: {'C': 0.005, 'dual': True, 'max_iter': 5000}--> F1: 0.88256
test F1: 0.88131
'''
def linearsvm(train_data, validation_data, test_data):
    print("Linear SVM")
    cv_train_data = np.hstack((train_data.T, validation_data.T)).T.astype('int')

    svm_cls = LinearSVC()

    parameters = {
        'dual': [True],
        'C': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'max_iter': [2000, 5000, 8000]
    }

    parameterTuning(svm_cls, parameters, cv_train_data[:, :-1], cv_train_data[:, -1])

    svm_cls = LinearSVC(dual=True, C=0.005, max_iter=5000)
    svm_cls.fit(cv_train_data[:, :-1], cv_train_data[:, -1])
    testClassifer(svm_cls, test_data[:, :-1].astype('int'), test_data[:, -1].astype('int'))

'''
max_depth --> controls overfitting, if none, all leafs are pure and can overfit. if overfitting, increase; otherwise decrease
min_samples_split --> (1 to 40 ideal), control overfitting. if overfit, increase this number, otherwise decrease
min_samples_leaf --> (1 to 20 ideal), control overfitting. if overfit increase, otherwise decrease
max_features --> used to reduce computation by considering less features at each split, and reduce overfitting

best params: {'max_depth': 2000, 'min_samples_leaf': 10, 'min_samples_split': 2} --> F1 score: 0.70656
test f1 score is  0.7049971835519433
'''
def decistionTree(train_data, validation_data, test_data):
    print("Decision Tree")
    cv_train_data = np.hstack((train_data.T, validation_data.T)).T
    dct_cls = DecisionTreeClassifier()
    
    parameters = {'max_depth' : [2000],
                  'min_samples_split' : [2],
                  'min_samples_leaf' : [1]
                  }

    parameterTuning(dct_cls, parameters, cv_train_data[:, :-1], cv_train_data[:, -1])
    dct_cls = DecisionTreeClassifier(max_depth=2000, min_samples_split=2, min_samples_leaf=1)
    dct_cls.fit(cv_train_data[:, :-1], cv_train_data[:, -1])
    testClassifer(dct_cls, test_data[:, :-1], test_data[:, -1])

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


def randomClassifier(x_train_data, y_train_data, x_test_data, y_test_data):
    print("Random Classifier")
    dumm_cls = DummyClassifier(strategy="uniform")
    dumm_cls.fit(x_train_data, y_train_data)

    predictions = dumm_cls.predict(x_test_data)
    evaluatePrediction(predictions, y_test_data)


def sckitiEvaluation(predictions, y_values):
    predictions = [int(x) for x in predictions]
    y_values = [int(x) for x in y_values]

    print("the scikit learn F1 score is: ", f1_score(y_values, predictions, average="binary"))
    print("============================================")

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


def fetchData(path):
    dataset = []
    with open(path, mode='r', encoding='utf-8') as data:
        data = data.readlines()
        for line in data:
            line = line.rstrip("\n")
            row = line.split('\t')

            review = row[0].split(' ')
            review.append(row[1])
            dataset.append(review)

    return np.array(dataset)


if __name__ == "__main__":
    main()