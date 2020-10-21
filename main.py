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
        diffItems = len(y) - (numCVItems + numTrainItems + numTestItems) # whatever i cant divide into cv and test, i put into train
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

    def costFunction(self, X, y, theta, lambd):
        m = len(y)
        sqrError = 0
        for i in range(0, m):
            sqrError += hypothesis
        return hypothesis







if __name__ == '__main__':
    csv = pd.read_csv("housing_prices.csv")
    predict = "Price"
    X = np.array(csv.drop(["Home", "Neighborhood", "Brick", predict], axis=1))
    y = np.array(csv[predict])


    brub = Main()
    # array = brub.trainTestSplit(X, y)
    #
    # xTrain = array[0]
    # yTrain = array[1]
    # xVal = array[2]
    # yVal = array[3]
    # xTest = array[4]
    # yTest = array[5]

    newArray = brub.costFunction(X, y, np.array([[0], [1]]), 1.0)
    print(newArray)


