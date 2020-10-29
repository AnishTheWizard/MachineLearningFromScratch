# create an algorithm from scratch, no libraries except numpy, pandas and other data management libraries allowed
import numpy as np
import pandas as pd
import DataManagement as dm


class Main():
    def __init__(self):
        pass

    def trainTestSplit(self, X, y) -> np.ndarray:
        numTrainItems = int(len(y) * 0.6)
        numCVItems = int(len(y) * 0.2)
        numTestItems = int(len(y) * 0.2)
        diffItems = len(y) - (
                numCVItems + numTrainItems + numTestItems)  # whatever i cant divide into cv and test, i put into train
        numTrainItems += diffItems

        print("starting dataSelection process")

        trainIndex = dm.dataSelection(numTrainItems, len(y))
        testIndex = dm.dataSelection(numTestItems, len(y))
        CVIndex = dm.dataSelection(numCVItems, len(y))

        print("preparing x vars")

        xTrain = np.asarray(dm.createDataset(X, trainIndex))
        xVal = np.asarray(dm.createDataset(X, CVIndex))
        xTest = np.asarray(dm.createDataset(X, testIndex))

        print("preparing y vars")

        yTrain = np.asarray(dm.createDataset(y, trainIndex))
        yVal = np.asarray(dm.createDataset(y, CVIndex))
        yTest = np.asarray(dm.createDataset(y, testIndex))

        print("Done!")

        return np.array([xTrain, yTrain, xVal, yVal, xTest, yTest], dtype=list)

    def costFunction(self, X, y, theta, lambd) -> float:
        m = len(X[0])  # number of features
        n = len(y)  # number of rows
        featureList = []
        for i in range(0, m):
            featureList.append(dm.appendColumns(X, i))

        hypothesis = theta[0][0] + np.multiply(theta[1][0], featureList[0]) + np.multiply(theta[2][0], featureList[1]) + np.multiply(theta[3][0], featureList[2]) + np.multiply(theta[4][0], featureList[3])
        sqrError = 0  # instantiating square error value
        for i in range(0, n):  # for each feature
                sqrError += (hypothesis[i] - y[i]) ** 2  # sqr error between prediction and y value

        sqrError *= 1 / (2 * n)  # taking the 1/2 of the average
        return sqrError  # return

if __name__ == '__main__':
    csv = pd.read_csv("housing_prices.csv")
    predict = "Price"
    X = np.array(csv.drop(["Home", "Neighborhood", "Brick", predict], axis=1))
    y = np.array(csv[predict])
    theta = np.array([[1], [1], [1], [1], [1]])

    brub = Main()
    # array = brub.trainTestSplit(X, y)
    #
    # xTrain = array[0]
    # yTrain = array[1]
    # xVal = array[2]
    # yVal = array[3]
    # xTest = array[4]
    # yTest = array[5]

    J = brub.costFunction(X, y, theta, 1.0)  # cost with given theta
    print(J)
