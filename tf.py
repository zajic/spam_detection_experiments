'''
this is just a learning experiment based on a tutorial, not exactly working yet
'''

import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix


# reset underlying graph data
tf.reset_default_graph()

data = pd.read_csv('../data/SMSSpamCollection.txt',header=None, sep="\t")

#map spam and ham values to 0 and 1
data.iloc[:,0] = data.iloc[:,0].map({'spam': 1, 'ham': 0})

count_vect = CountVectorizer()
data_transformed = count_vect.fit_transform(data[1])

X_train, X_test, y_train, y_test = train_test_split(data_transformed, data[0], test_size=0.3)

# Build neural network
net = tflearn.input_data(shape=[None,X_train.shape[1]])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(y_train), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

#convert sparse matrix to dense
X = csr_matrix.todense(X_train)

#train model
model.fit(X, y_train, n_epoch=100, batch_size = 8, show_metric=True)
model.save('model.tflearn')