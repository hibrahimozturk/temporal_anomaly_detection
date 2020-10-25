from feature_extract.input_process.InputOutput import InputOutput
import queue
import copy
from PIL import Image

import logging
logger = logging.getLogger('extractor')


class TSMInput(InputOutput):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict,
                 inputLength: int, batchSize: int, inputSize=(224, 224)):
        temporalSlide = 64 // inputLength
        InputOutput.__init__(self, batch, videoPath, annotations, temporalSlide,
                             inputLength, batchSize, inputSize)

        self.frameTargets = []
        self.featureNames = []

    def prepare_frame(self, frame):
        frame = Image.fromarray(frame)
        return frame

    def prepare_input(self, fps):
        self.__is_abnormal(fps)
        if len(self.frames) == self.inputLength:
            self.inputClips.append(copy.deepcopy(self.frames))
            self.clipNames.append(copy.deepcopy(self.featureNames))
            self.videoNames.append([self.videoName] * self.inputLength)
            self.targets.append(copy.deepcopy(self.frameTargets))
            self.frames, self.featureNames, self.frameTargets = [], [], []
        return

    def __is_abnormal(self, fps):
        anomaly = 0
        for actionSpace in self.annotations:
            if actionSpace["start"] * fps < self.frameCounter - 1 < actionSpace["end"] * fps:
                anomaly = 1
                break
            else:
                anomaly = 0
        self.frameTargets.append(anomaly)
        self.featureNames.append(self.videoName + "_" + str(self.frameCounter - 1).zfill(10))
