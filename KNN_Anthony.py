import math
import operator
import random
import pandas as pd
import numpy as np

from classifier import classifier

class knn(classifier):
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X, Y):
        # noop
        pass

    def predict(self, testX, train, test):
        predictions = []
        for x in range(len(testX)):
            neighbors = self.getNeighbors(train, test[x][0])
            majority = self.getMajorityClass(neighbors)
            predictions.append(majority)
            # print('predicted label = ' + str(majority) + ', actual label = ' + str(test[x][1]))
        accuracy = self.get_accuracy(test, predictions)
#         print('Accuracy is: ' + str(accuracy))
        return str(accuracy)

    # get neighbors according to its distance
    def getNeighbors(self, trainingSet, testData):
        distances = []
        for trainData in trainingSet:
            dist = self.getDistance(trainData[0], testData)
            distances.append((trainData, dist))
        # order by distance
        distances = sorted(distances, key=operator.itemgetter(1))
        sortedTraindata = [tuple[0] for tuple in distances]
        # pick up top k elements
        kNeighbors = sortedTraindata[:self.k]
        return kNeighbors

    # get euclidean distance
    def getDistance(self, instance1, instance2):
        pairs = zip(instance1, instance2)
        diffsSquared = 0
        for x1, x2 in pairs:
            diffsSquared += pow((x1 - x2), 2)
        return math.sqrt(diffsSquared)

    # calculate out the class of neighbors and decide the class by the frequency
    def getMajorityClass(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse = True)
        return sortedVotes[0][0]

    def get_accuracy(self, test_set, predictions):
        correct = 0
        for x in range(len(test_set)):
            if test_set[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(test_set)))



from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff('PhishingData.arff')
df = pd.DataFrame(data)
df = df.convert_objects(convert_numeric=True)

dataset = df.values.tolist()
splitpoint = int(len(dataset)*0.8)
train = dataset[:splitpoint+1]
test = dataset[splitpoint:]

x = df.iloc[:, 0:9]
y = df.iloc[:, 9]

xArr = x.values
yArr = y.values

split = int(len(yArr) * 0.8)
# training data
trainX = xArr[:split+1]
trainY = yArr[:split+1]
# test data
testX = xArr[split:]
testY = yArr[split:]

# reform datasets
train = list(zip(trainX, trainY))
test = list(zip(testX, testY))

print('K\tAccuracy')
for k in range(2, 33):
    knnInstance = knn(k)
    accuracy = knnInstance.predict(testX,train,test)
    print(str(k) + '\t' + accuracy)
