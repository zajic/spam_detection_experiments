"""
train and evaluate random forests classifier
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from my_metrics import calculate_metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

clf = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', RandomForestClassifier()),
 ])

clf_noTfidf = Pipeline([('vect', CountVectorizer()),
                ('clf', RandomForestClassifier()),
 ])

data = pd.read_csv('../data/SMSSpamCollection.txt',header=None, sep="\t")

train, test = train_test_split(data, test_size=0.2)
Y_train = train[0]
X_train = train[1]

clf.fit(X_train, Y_train)

print("\nRandom forests with TFIDF\n")
clf.fit(X_train, Y_train)

Y_pred = clf.predict(test[1])
Y_actual = test[0]

calculate_metrics(Y_actual,Y_pred)

print("\nRandom forests NO TFIDF\n")
clf_noTfidf.fit(X_train, Y_train)

Y_pred = clf_noTfidf.predict(test[1])
Y_actual = test[0]

calculate_metrics(Y_actual,Y_pred)


