"""
Train and evaluate logistic regression classifier.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from my_metrics import calculate_metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
 ])

clf_noTfidf = Pipeline([('vect', CountVectorizer()),
                ('clf', LogisticRegression()),
 ])

data = pd.read_csv('../data/SMSSpamCollection.txt',header=None, sep="\t")

train, test = train_test_split(data, test_size=0.2)

Y_train = train[0]
X_train = train[1]

clf.fit(X_train, Y_train)

print("\nLogistic Regression with TFIDF\n")
clf.fit(X_train, Y_train)

Y_pred = clf.predict(test[1])
Y_actual = test[0]

calculate_metrics(Y_actual,Y_pred)

print("\nLogistic Regression NO TFIDF\n")
clf_noTfidf.fit(X_train, Y_train)

Y_pred = clf_noTfidf.predict(test[1])
Y_actual = test[0]

calculate_metrics(Y_actual,Y_pred)







