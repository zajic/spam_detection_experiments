"""
Trains a Naive Bayes classifier and saves to pickle.
Saved classifier can be used to predict previously unseen data.

"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from my_metrics import calculate_metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import glob
import os
import time

def train_model():

	#TODO: Grid Search
    clf_noTfidf = Pipeline([('vect', CountVectorizer()),
                    ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
     ])

    data = pd.read_csv('../data/SMSSpamCollection.txt', header=None, sep="\t")

    Y = data[0]
    X = data[1]

    clf_noTfidf.fit(X, Y)

    #save trained classifier to pickle for future use
    with open('nb_classifier', 'wb') as fp:
        pickle.dump(clf_noTfidf, fp)

#find the newest file in the current directory and uses it as input to predict spam/ham
def predict_unseen_data():

    #get latest file
    list_of_files = glob.glob(os.getcwd() + "/file*")
    latest_file = max(list_of_files, key=os.path.getctime)

    #load data
    with open(latest_file, 'rb') as fp:
        data = pickle.load(fp)

    #load classifier
    with open('nb_classifier', 'rb') as fp:
        clf = pickle.load(fp)


    pred_data = pd.DataFrame(clf.predict(data.iloc[:,0]))
    data = pd.concat([pred_data,data],axis=1)
    filename = "./data/predicted_" + str(int(time.time()))
    data.to_csv(filename + '.csv', sep="\t", index=False)

    print(data)

train_model()
# predict_unseen_data()