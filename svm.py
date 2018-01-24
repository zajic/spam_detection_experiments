"""
Train a SVM classifier and evaluate model metrics
"""
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from my_metrics import calculate_metrics

#TODO: Grid Search
clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])

clf_noTfidf = Pipeline([('vect', CountVectorizer()),
                        ('clf',SGDClassifier(loss='hinge', penalty='l2',
                                      alpha=1e-3, random_state=42,
                                      max_iter=5, tol=None)),
])

data = pd.read_csv('../data/SMSSpamCollection.txt',header=None, sep="\t")

train, test = train_test_split(data, test_size=0.2)
Y_train = train[0]
X_train = train[1]

clf.fit(X_train, Y_train)

print("\nSupport Vector Machines with TFIDF\n")
clf.fit(X_train, Y_train)

Y_pred = clf.predict(test[1])
Y_actual = test[0]

calculate_metrics(Y_actual,Y_pred)

print("\nSupport Vector Machines NO TFIDF\n")
clf_noTfidf.fit(X_train, Y_train)

Y_pred = clf_noTfidf.predict(test[1])
Y_actual = test[0]

calculate_metrics(Y_actual,Y_pred)