import copy

from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
import torch
from addict import Dict

import logging
logger = logging.getLogger("violance")


def get_dataloaders(trainNormalFolder: str, trainNormalAnnotations: str,
                    trainAbnormalFolder: str, trainAbnormalAnnotations: str,
                    trainTopK: int,
                    valNormalFolder: str, valNormalAnnotations: str,
                    valAbnormalFolder: str, valAbnormalAnnotations: str,
                    valTopK: int,
                    batchSize: int, numWorkers: int,
                    model: str, windowSize: int, subWindow: int, featureSize: int,
                    maxVideoSize: int):
    if model == "mlp":
        trainDataset = UCFCrimeDataset(trainAbnormalFolder, trainAbnormalAnnotations,
                                       trainNormalFolder, trainNormalAnnotations)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True,
                                 num_workers=numWorkers, pin_memory=True)
    elif model == "tcn" or model == "mstcn" or model == "mcbtcn":
        trainDataset = UCFCrimeTemporal(trainAbnormalFolder, trainAbnormalAnnotations,
                                        trainNormalFolder, trainNormalAnnotations,
                                        windowSize=windowSize, normalTopK=trainTopK,
                                        subWindow=subWindow, featureSize=featureSize,
                                        maxVideoSize=maxVideoSize)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers,
                                 collate_fn=collate_fn_precomp, pin_memory=True)

    if model == "mlp":
        valDataset = UCFCrimeDataset(valAbnormalFolder, valAbnormalAnnotations,
                                     valNormalFolder, valNormalAnnotations)
        valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
    elif model == "tcn" or model == "mstcn" or model == "mcbtcn":
        valDataset = UCFCrimeTemporal(valAbnormalFolder, valAbnormalAnnotations,
                                      valNormalFolder, valNormalAnnotations,
                                      windowSize=windowSize, normalTopK=valTopK,
                                      subWindow=subWindow, featureSize=featureSize,
                                      maxVideoSize=maxVideoSize)

        valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers,
                               collate_fn=collate_fn_precomp)

    return trainLoader, valLoader


class UCFCrimeDataset(Dataset):
    def __init__(self, data_cfg, split):

        split_cfg = getattr(data_cfg, split)
        with open(split_cfg.abnormal.annotations) as fp:
            self.abnormal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.abnormal.clip_features,
                top_k=split_cfg.abnormal.top_k if hasattr(split_cfg.abnormal, "top_k") else None
            ))
        with open(split_cfg.normal.annotations) as fp:
            self.normal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.normal.clip_features,
                top_k=split_cfg.abnormal.top_k if hasattr(split_cfg.abnormal, "top_k") else None
            ))

        self.clips = []
        self.clipLists = {}

        for clipName, value in self.abnormal.annotations["all_clips"].items():
            self.clips.append({"path": os.path.join(self.abnormal.clip_features, clipName + ".npy"),
                               "anomaly": value["anomaly"],
                               "category": value["category"],
                               "category_name": value["category_name"],
                               "clip_name": clipName})

        for clipName, value in self.normal.annotations["all_clips"].items():
            self.clips.append({"path": os.path.join(self.normal.clip_features, clipName + ".npy"),
                               "anomaly": value["anomaly"],
                               "category": value["category"],
                               "category_name": value["category_name"],
                               "clip_name": clipName})

        for videoIndex, videoName in enumerate(self.abnormal.annotations["video_clips"]):
            self.clipLists[videoName] = self.abnormal.annotations["video_clips"][videoName]

        for videoIndex, videoName in enumerate(self.normal.annotations["video_clips"]):
            self.clipLists[videoName] = self.normal.annotations["video_clips"][videoName]

        print("Dataset has been constructed")
        print("# abnormal clips: {}".format(len(self.abnormal.annotations["abnormal_clips"])))
        print("# normal clips: {}".format(len(self.abnormal.annotations["normal_clips"])))

    def __len__(self):
        return len(self.clips)

    def __getVideoClips__(self):
        return self.clipLists

    def __getitem__(self, idx):
        annotation = self.clips[idx]
        feature = np.load(os.path.join(annotation["path"]))

        return feature, 1, annotation["anomaly"], annotation["category"], \
               annotation["category_name"], annotation["clip_name"]


class UCFCrimeTemporal(Dataset):
    def __init__(self, data_cfg, split):

        split_cfg = getattr(data_cfg, split)
        with open(split_cfg.abnormal.annotations) as fp:
            self.abnormal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.abnormal.clip_features,
                top_k=split_cfg.abnormal.top_k if hasattr(split_cfg.abnormal, "top_k") else None
            ))
        with open(split_cfg.normal.annotations) as fp:
            self.normal = Dict(dict(
                annotations=json.load(fp),
                clip_features=split_cfg.normal.clip_features,
                top_k=split_cfg.abnormal.top_k if hasattr(split_cfg.abnormal, "top_k") else None
            ))

        self.clips = {}
        for part in [self.abnormal, self.normal]:
            for clip_name, value in part.annotations.all_clips.items():
                self.clips[clip_name] = {"path": os.path.join(part.clip_features, clip_name + ".npy"),
                                         "anomaly": value["anomaly"],
                                         "category": value["category"],
                                         "category_name": value["category_name"]}

        self.window_size = data_cfg.window_size
        self.featureSize = data_cfg.feature_size
        self.sub_window_size = int(self.window_size / data_cfg.sub_windows)
        self.mask_value = data_cfg.mask_value
        self.clipLists = {}
        self.windows = []

        total_windows = 0
        for part_name, part in dict(normal=self.normal, abnormal=self.abnormal).items():
            num_clips = self.__prepare_windows(part.annotations, maxVideoSize=data_cfg.max_video_len, topK=part.top_k)
            part.num_clips = num_clips
            logger.info("# of {} windows: {}".format(part_name, len(self.windows)-total_windows))
            total_windows += len(self.windows)
            if hasattr(part, "top_k"):
                logger.info("first {} {} videos have been included".format(part.top_k, part_name))

        # Abnormal clips in only abnormal videos
        # Normal clips are in normal and abnormal videos
        logger.info("# abnormal clips: {}".format(self.abnormal.num_clips))
        logger.info("# normal clips: {}".format(self.abnormal.num_clips + self.normal.num_clips))
        logger.info("dataset has been constructed")

    def __prepare_windows(self, annotations, maxVideoSize=None, topK=None):
        videoNames = list(annotations["video_clips"].keys())
        videoNames.sort()
        if topK is not None:
            videoNames = videoNames[:topK]
        clipLengths = np.array([len(annotations["video_clips"][key]) for key in videoNames])
        # print("[Info] Max clip length for dataset is {}".format(np.max(clipLengths)))
        if maxVideoSize is not None:
            filteredClips = np.where(clipLengths < maxVideoSize)[0]
            clipLengths = clipLengths[filteredClips]
            videoNames = [videoName for i, videoName in enumerate(videoNames) if i in filteredClips]
        numOfClips = np.sum(clipLengths)
        videoWindows = np.ceil(clipLengths / self.sub_window_size)
        for videoIndex, videoName in enumerate(videoNames):
            self.clipLists[videoName] = annotations["video_clips"][videoName]
            videoClipList = copy.deepcopy(annotations["video_clips"][videoName])
            for clipIndex in range(len(videoClipList),
                                   max(int(videoWindows[videoIndex] * self.sub_window_size), self.window_size)):
                videoClipList.append("")

            for windowIndex in range(max(int(videoWindows[videoIndex]) - 1, 1)):
                start = windowIndex * self.sub_window_size
                window = videoClipList[start:start + self.window_size]
                assert len(window) == self.window_size, "window size does not match"
                self.windows.append(window)
        return numOfClips

    def __len__(self):
        return len(self.windows)

    def __getVideoClips__(self):
        return self.clipLists

    def __getitem__(self, item):
        windowClipNames = self.windows[item]
        inputData = np.zeros((self.window_size, self.featureSize))
        anomalies = []
        categories = []
        categoryNames = []
        masks = np.zeros((self.window_size, 1))
        for index, clipName in enumerate(windowClipNames):
            if clipName != "":
                clip = np.load(self.clips[clipName]["path"])
                mask = 1
                # annotation = self.abnormalAnnotations["allClips"][clipName]
                anomalies.append(self.clips[clipName]["anomaly"])
                categories.append(self.clips[clipName]["category"])
                categoryNames.append((self.clips[clipName]["category_name"]))
            else:
                clip = np.zeros(self.featureSize) + self.mask_value
                mask = 0
                anomalies.append(-1)
                categories.append(-1)
                categoryNames.append("")
            inputData[index, :] = clip
            masks[index, :] = mask
        inputData = torch.from_numpy(inputData)
        anomalies = torch.from_numpy(np.array(anomalies))

        # abnormal video normal clips classified as normal (15)
        categoriesMerged = np.ones_like(anomalies) * 15
        categories = np.array(categories)
        categoriesMerged[anomalies == 1] = categories[anomalies == 1]

        categories = torch.from_numpy(np.array(categories))
        masks = torch.from_numpy(np.array(masks))
        return inputData, masks, anomalies, categories, categoryNames, windowClipNames


def collate_fn_precomp(data):
    inputData, masks, anomalies, categories, categoryNames, windowClipNames = zip(*data)

    inputData = torch.stack(inputData, 0)
    anomalies = torch.stack(anomalies, 0)
    masks = torch.stack(masks, 0)
    categories = torch.stack(categories, 0)

    return inputData, masks, anomalies, categories, categoryNames, windowClipNames


if __name__ == "__main__":
    valSet = UCFCrimeTemporal("../data/i3d_features/abnormal/train",
                              "../data/i3d_features/abnormal/TrainLabels.json",
                              "../data/i3d_features/normal/train",
                              "../data/i3d_features/normal/TrainLabels.json",
                              normalTopK=100)
    dataloader = DataLoader(valSet, batch_size=16, shuffle=True, num_workers=20, collate_fn=collate_fn_precomp)
    for i, batch in enumerate(tqdm(dataloader)):
        i
