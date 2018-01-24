"""
train and evaluate performance of 4 different models: RF, SVM, NB and logistic regression
* compare performance of models with and without performing TFIDF weighting
* using 5-fold cross-validation
* compare accuracy and false negative rate
* each time the code is run a different random seed for CV train/test split is used but this can be adjusted in ShuffleSplit()
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
import random
import numpy as np

clf_nb = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
 ])

clf_nb_noTfidf = Pipeline([('vect', CountVectorizer()),
                ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
 ])

clf_rf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier()),
 ])

clf_rf_noTfidf = Pipeline([('vect', CountVectorizer()),
                ('clf', RandomForestClassifier()),
 ])

clf_lr = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
 ])

clf_lr_noTfidf = Pipeline([('vect', CountVectorizer()),
                ('clf', LogisticRegression()),
 ])

clf_svm = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])

clf_svm_noTfidf = Pipeline([('vect', CountVectorizer()),
                        ('clf',SGDClassifier(loss='hinge', penalty='l2',
                                      alpha=1e-3, random_state=42,
                                      max_iter=5, tol=None)),
])

data = pd.read_csv('../data/SMSSpamCollection.txt', header=None, sep="\t")

'''
create a custom scoring function for FNR (false negative rate)
(cross validation can return accuracy scores and other metrics but we want to evaluate FNR score)
'''
def fnr_scoring(actual,pred):
    TN, FP, FN, TP = confusion_matrix(actual, pred).ravel()
    FNR = round(FN / (FN + TP), 2)
    return FNR

#this will be passed as parameter to cross validation
my_scoring = make_scorer(fnr_scoring,greater_is_better=True)
seed = random.randint(1,100)

#define parameters for cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=seed)
classifiers = {"clf_nb":clf_nb,"clf_nb_noTfidf":clf_nb_noTfidf,"clf_rf":clf_rf,"clf_rf_noTfidf":clf_rf_noTfidf,
                    "clf_lr":clf_lr,"clf_lr":clf_lr,"clf_svm":clf_svm,"clf_svm_noTfidf":clf_svm_noTfidf}

def cross_validate(clf,clf_name,seed,cv,my_scoring):


    print(clf_name + "\n")

    #perform 5-fold cross-validation with FNR scoring metric
    scores = cross_val_score(clf, data[1], data[0], cv=cv, scoring = my_scoring)
    scores = np.array([round(x,2) for x in scores])
    print("FNR")
    print(scores)
    print("average: " + str(scores.mean()) + "\n")

    #perform 5-fold cross-validation with ACCURACY scoring metric
    scores = cross_val_score(clf, data[1], data[0], cv=cv)
    scores = np.array([round(x,2) for x in scores])
    print("Accuracy")
    print(scores)
    print("average: " + str(scores.mean()))
    print("\n")


for clf in classifiers:
    cross_validate(classifiers[clf],clf,seed,cv,my_scoring)


