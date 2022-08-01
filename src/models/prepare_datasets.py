import os
import shutil
import random


def transferBetweenFolders(source, dest, splitRate):
    global sourceFiles
    sourceFiles = os.listdir(source)
    if (len(sourceFiles) != 0):
        transferFileNumbers = int(len(sourceFiles) * splitRate)
        transferIndex = random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source + str(sourceFiles[eachIndex]), dest + str(sourceFiles[eachIndex]))
    else:
        print("No file moved. Source empty!")


def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName + '/' + source + '/' + label + '/',
                               datasetFolderName + '/' + dest + '/' + label + '/',
                               splitRate)


def prepareNameWithLabels(folderName):
    sourceFiles=os.listdir(datasetFolderName+'/train/'+folderName)
    for val in sourceFiles:
        X.append(val)
        for i in range(len(classLabels)):
            if(folderName==classLabels[i]):
                Y.append(i)


