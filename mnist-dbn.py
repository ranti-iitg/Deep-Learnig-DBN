from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN

#automatic data-fetching
dataset = datasets.fetch_mldata("MNIST Original")

#load data set usng train_test_split note that data scale is [0-1.0] , test size is 1/3 of total data
(trainX, testX, trainY, testY) = train_test_split(
dataset.data / 255.0, dataset.target.astype("int0"), test_size = 0.33)

#DBN having 3 hidden layer having 300 hidden units learnrate 0.1
dbn = DBN(
[trainX.shape[1], 300,300,300, 10],
learn_rates = 0.1,
learn_rate_decays = 0.9,
epochs = 20,
verbose = 1)
dbn.fit(trainX, trainY)

preds = dbn.predict(testX)
print classification_report(testY, preds)