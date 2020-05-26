# ML

## NLP

This folder contains NLP algorithms that classify the IMDB reviews as positive or negative. Regular ML methods are used, such as Logistic Regression, Naive Bayes, Linear SVM, and Decision Trees, as well as deep learning methods, such as Neural Nets and CNN.

1. Binary Bag of Words (BBOW)

The IMDB dataset is vectorized using sklearn's CountVectorizer. The vocab and new datasets are stored in files. In this section, only the ML methods are tuned, the vectorizer is not tuned (we will tune it in the next section).

- Naive Bayes:
Achieved a test F1 score of 0.684 with parameters: var_smoothing=5e-7 

- Linear SVM:
Achieved a test F1 score of 0.881 wiht parameters: {'C': 0.005, 'dual': True, 'max_iter': 5000}

- Decision Tree:
Achieved a test F1 score of 0.705 with parameters: {'max_depth': 2000, 'min_samples_leaf': 10, 'min_samples_split': 2}

- Logistic Regression:



2. Frequency Bag of Words (FBOW)
The IMDB dataset is vectorized using sklearn's CountVectorizer. The vocab and new datasets are stored in files. In this section, only the ML methods are tuned, the vectorizer is not tuned (we will tune it in the next section).
The difference with 1., is that here we use the frequency of words to vectorize our dataset, instead of just binary.

- Naive Bayes:
Achieved a test F1 score of 0.654 with parameters: var_smoothing=1e-6

- Linear SVM:
Achieved a test F1 score of 0.875 wiht parameters: {'alpha': 0.001, 'max_iter': 2000}

- Decision Tree:
Achieved a test F1 score of 0.708 with parameters: {'max_depth': 4000, 'min_samples_leaf': 10, 'min_samples_split': 2}

- Logistic Regression
Achieved a test F1 score of 0.886 wiht parameters: {'alpha': 0.001, 'loss': 'log', 'max_iter': 100, 'n_jobs': 4}


3. Tuned BOW
I used the sklearn pipeline and GridSearchCV, to tune the bag of words along with the machine learning methods. I have used the SGD classifier to model 

4. Neural Network with FBOW


5. Word Embeddings with CNN
