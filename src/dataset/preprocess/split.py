import math

import numpy as np
import json


def main():
    trainJson = "../../data/Annotations/TemporalAnnotations/abnormal/train.json"
    testJson = "../../data/Annotations/TemporalAnnotations/abnormal/test.json"
    testVideosList = "../../data/Annotations/Anomaly_Detection_splits/Anomaly_Test.txt"
    excludeNormal = True
    excludeAbnormal = False

    fixWrongSplit = True

    validationRatio = 0.1

    with open(trainJson, "r") as fp:
        trainAnnotations = json.load(fp)

    testList = readTestFiles(testVideosList, excludeNormal=excludeNormal)
    with open(testJson, "r") as fp:
        testAnnotations = json.load(fp)

    temporalAnnotations = {**trainAnnotations, **testAnnotations}
    print("Num of total temporal annotations: {}".format(len(temporalAnnotations)))

    # Eliminate normal videos
    newAnnotations = {}
    for key, value in temporalAnnotations.items():
        if excludeNormal and "Normal_Videos" not in key:
            newAnnotations[key] = temporalAnnotations[key]
        elif excludeAbnormal and "Normal_Videos" in key:
            newAnnotations[key] = temporalAnnotations[key]

    newTestAnnotations = {}
    for testVideo in testList:
        if excludeNormal and "Normal_Videos" not in testVideo:
            newTestAnnotations[testVideo] = newAnnotations[testVideo]
            del(newAnnotations[testVideo])
        elif excludeAbnormal and "Normal_Videos" in testVideo:
            newTestAnnotations[testVideo] = newAnnotations[testVideo]
            del(newAnnotations[testVideo])

    keyList = np.array([x for x in newAnnotations.keys()])
    indexList = np.arange(len(keyList))
    np.random.shuffle(indexList)

    trainIndexes = indexList[:math.floor(len(indexList)*(1-validationRatio))]
    valIndexes = indexList[math.floor(len(indexList)*(1-validationRatio)):]

    if fixWrongSplit:

        with open("../../data/i3d_features/abnormal/ValLabels.json", "r") as fp:
            valJson = json.load(fp)

        with open("../../data/Annotations/Splits/abnormal/val.json", "r") as fp:
            valSplit = json.load(fp)

        with open("../../data/i3d_features/abnormal/TrainLabels.json", "r") as fp:
            trainJson = json.load(fp)

        with open("../../data/Annotations/Splits/abnormal/train.json", "r") as fp:
            trainSplit = json.load(fp)

        trainIndexes = []
        # for videoName in list(trainJson["videoClips"].keys()):
        for videoName in list(trainSplit.keys()):
            counter = 0
            for i, videoN in enumerate(keyList):
                if videoName in videoN:
                    counter += 1
                    trainIndexes.append(i)
                    if counter > 1:
                        print("Error!!! Duplicate")
        trainIndexes = np.array(trainIndexes)

        valIndexes = []
        # for videoName in list(valJson["videoClips"].keys()):
        for videoName in list(valSplit.keys()):
            for i, videoN in enumerate(keyList):
                if videoName in videoN:
                    valIndexes.append(i)
        valIndexes = np.array(valIndexes)

    trainKeys = keyList[trainIndexes]
    valKeys = keyList[valIndexes]

    valAnnotations = {}
    for key in valKeys:
        valAnnotations[key] = newAnnotations[key]
        del(newAnnotations[key])

    trainAnnotations = {}
    for key in trainKeys:
        trainAnnotations[key] = newAnnotations[key]
        del(newAnnotations[key])

    if fixWrongSplit:
        restKeys = list(newAnnotations.keys())
        for key in restKeys:
            trainAnnotations[key] = newAnnotations[key]
            del(newAnnotations[key])

        valVideoNames = list(valSplit.keys())
        includedIndexes = []
        for i, videoName in enumerate(valVideoNames):
            include = True
            for extractedVideoName in list(valAnnotations.keys()):
                if extractedVideoName in videoName:
                    include = False
            if include:
                includedIndexes.append(i)
        print("Omitted validation videos in feature extraction:")
        print(len([valVideoNames[i] for i in includedIndexes]))
        print([valVideoNames[i] for i in includedIndexes])

    assert(len(newAnnotations) == 0)

    with open("test.json", "w") as fp:
        json.dump(newTestAnnotations, fp)

    with open("train.json", "w") as fp:
        json.dump(trainAnnotations, fp)

    with open("val.json", "w") as fp:
        json.dump(valAnnotations, fp)

    print("# Videos in train set: {}".format(trainIndexes.shape[0]))
    print("# Videos in val set: {}".format(valIndexes.shape[0]))
    print("# Videos in test set: {}".format(len(newTestAnnotations)))

    return


def readTestFiles(testVideosList, excludeNormal=False):
    testList = []
    with open(testVideosList, "r") as fp:
        for testVideo in fp:
            if excludeNormal and "Normal_Videos" in testVideo:
                continue
            testList.append(testVideo.strip())
    return testList


if __name__ == "__main__":
    main()
