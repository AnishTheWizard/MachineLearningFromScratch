import numpy as np
import random as rand


def dataSelection(numOfItems, lengthOfY) -> list:
    array = []
    for i in range(0, numOfItems):
        isFound = True
        while isFound:  # until you find an index that is not already in the list
            randomIndex = rand.randrange(0, lengthOfY)  # create a random index to test
            isFound = doesTheItemExist(randomIndex, array)
        array.append(randomIndex)
    return array


def createDataset(dataset, indexList) -> list:
    returnList = []
    for item in indexList:
        returnList.append(np.array([item, dataset[item]], dtype=list))
    return returnList


def doesTheItemExist(item, testArray) -> bool:
    falseUntilTrue = False
    for thing in testArray:
        if item == thing:
            falseUntilTrue = True

    return falseUntilTrue


