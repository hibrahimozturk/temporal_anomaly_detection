import argparse
import json


def params():
    parser = argparse.ArgumentParser(description='Anomaly Detection in Videos')
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--model", type=str, default="tcn", help="mlp | tcn")
    parser.add_argument("--loss", type=str, default="tcn", help="mlp | tcn | ad")
    parser.add_argument("--adLossLambda", type=float, default=None)
    parser.add_argument("--adLossMargin", type=float, default=None)

    parser.add_argument("--featureSize", type=int, default=1024, help="size of extracted feature")
    parser.add_argument("--numFeatureMaps", type=int, default=64, help="size of extracted feature")
    parser.add_argument("--kernelSize", type=int, default=None, help="size of TCN filter kernel size")
    parser.add_argument("--windowSize", type=int, default=20)
    parser.add_argument("--maxVideoSize", type=int, default=None)
    parser.add_argument("--subWindows", type=int, default=2)
    parser.add_argument("--maskValue", type=int, default=-1)

    # MS-TCN
    parser.add_argument("--numStages", type=int, default=1)

    # MCB-TCN
    parser.add_argument("--numClassStages", type=int, default=1)
    parser.add_argument("--numBinaryStages", type=int, default=1)

    parser.add_argument("--firstStageRepeat", type=int, default=1)
    parser.add_argument("--consecutiveStagesRepeat", type=int, default=1)
    parser.add_argument("--numLayers", type=int, default=1)

    parser.add_argument("--trainNormalFolder", type=str,
                        default="../data/ucfcrime_features/normal/train")
    parser.add_argument("--trainNormalAnnotations", type=str,
                        default="../data/ucfcrime_features/normal/TrainLabels.json")
    parser.add_argument("--trainNormalTopK", type=int)

    parser.add_argument("--trainAbnormalFolder", type=str,
                        default="../data/ucfcrime_features/abnormal/train")
    parser.add_argument("--trainAbnormalAnnotations", type=str,
                        default="../data/ucfcrime_features/abnormal/TrainLabels.json")

    parser.add_argument("--valNormalFolder", type=str,
                        default="../data/ucfcrime_features/normal/val")
    parser.add_argument("--valNormalAnnotations", type=str,
                        default="../data/ucfcrime_features/normal/ValLabels.json")
    parser.add_argument("--valNormalTopK", type=int)

    parser.add_argument("--valAbnormalFolder", type=str,
                        default="../data/ucfcrime_features/abnormal/val")
    parser.add_argument("--valAbnormalAnnotations", type=str,
                        default="../data/ucfcrime_features/abnormal/ValLabels.json")

    # parser.add_argument("--testFolder", type=str)
    # parser.add_argument("--testAnnotations", type=str)
    # parser.add_argument("--testNormalTopK", type=int)

    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batchSize", type=int, default=16)
    parser.add_argument("--numWorkers", type=int, default=4)
    parser.add_argument("--schedulerStepSize", type=int, default=None)
    parser.add_argument("--schedulerGamma", type=float, default=None)
    parser.add_argument("--learningRate", type=float, default=0.0001)
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--expFolder", type=str, default="../exps/1-window_20")
    parser.add_argument("--modelPath", type=str)

    parser.add_argument("--noNormalSegmentation", action="store_true")

    parser.add_argument("--json-conf", type=str)

    return parser.parse_args()


def parseJson(args, jsonPath):
    with open(jsonPath, "r") as fp:
        jsonConfs = json.load(fp)

    for key, value in jsonConfs.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            assert False, "the key is not in added arguments " + key
    return args


args = params()
if args.json_conf is not None:
    args = parseJson(args, args.json_conf)
print(args)
