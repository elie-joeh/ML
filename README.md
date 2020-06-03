# ML

## NLP

This folder contains NLP algorithms that classify the IMDB reviews as positive or negative. Regular ML methods are used, such as Logistic Regression, Naive Bayes, Linear SVM, and Decision Trees, as well as deep learning methods, such as Neural Nets and CNN.

1. **Binary Bag of Words (BBOW)**

The IMDB dataset is vectorized using sklearn's CountVectorizer. The vocab and new datasets are stored in files. In this section, only the ML methods are tuned, the vectorizer is not tuned (we will tune it in the next section).

- Naive Bayes:

Achieved a test F1 score of **0.684** with parameters: **var_smoothing=5e-7** 

- Linear SVM:

Achieved a test F1 score of **0.881** wiht parameters: **{'C': 0.005, 'dual': True, 'max_iter': 5000}**

- Decision Tree:

Achieved a test F1 score of **0.705** with parameters: **{'max_depth': 2000, 'min_samples_leaf': 10, 'min_samples_split': 2}**

- Logistic Regression:



2. **Frequency Bag of Words (FBOW)**

The IMDB dataset is vectorized using sklearn's CountVectorizer. The vocab and new datasets are stored in files. In this section, only the ML methods are tuned, the vectorizer is not tuned (we will tune it in the next section).
The difference with 1., is that here we use the frequency of words to vectorize our dataset, instead of just binary.

- Naive Bayes:

Achieved a test F1 score of **0.654** with parameters: **var_smoothing=1e-6**

- Linear SVM:

Achieved a test F1 score of **0.875** wiht parameters: **{'alpha': 0.001, 'max_iter': 2000}**

- Decision Tree:

Achieved a test F1 score of **0.708** with parameters: **{'max_depth': 4000, 'min_samples_leaf': 10, 'min_samples_split': 2}**

- Logistic Regression

Achieved a test F1 score of **0.886** wiht parameters: **{'alpha': 0.001, 'loss': 'log', 'max_iter': 100, 'n_jobs': 4}**


3. **Tuned BOW**

I used the sklearn pipeline and GridSearchCV, to tune the bag of words along with the machine learning methods. I have used the SGD classifier to model.

- Linear SVML

Achieved a test F1 score of **0.902** with the parameter set:
	clf__alpha: 1e-05
	clf__max_iter: 100
	vect__max_df: 0.3
	vect__max_features: None
	vect__ngram_range: (1, 2)

- Logistic Regression

Achieved a test F1 score of **0.902** with the parameter set:
	clf__alpha: 1e-06
	clf__eta0: 0.001
	clf__learning_rate: 'optimal'
	clf__loss: 'log'
	clf__shuffle: True
	vect__max_df: 0.2
	vect__max_features: None
	vect__ngram_range: (1, 2)


4. **Neural Network with FBOW**

After hypertuning the NN, the best prediction reached is **88.7%** with 3 hidden layers, with 50, 25 and 10 nodes respectively.
The optimal parameters are: **{'batch_size': 1000, 'dropout_rate': 0.8, 'epochs': 30, 'weight_decay': 1e-06}**



5. **CNN Word Embeddings**

In this section, I have used the google pretrained word2vec embedding. I created a CNN with an embedding layer and hypertuned it with 1, 2, 3 and 4 hidden layers. I used Gridsearch and KerasClassifier to efficiently hypertune the network. I was able to reach an accuracy of **0.80** with 4 convolutional hidden layers, with 128 nodes each, batchnorm maxpooling and dropout after each layer, and with adam optimzation. The following parameters gave the best performance: **{'batch_size': 90, 'epochs': 30, 'init': 'glorot_uniform', 'kernel_size': 2, 'pool_size': 5}**
