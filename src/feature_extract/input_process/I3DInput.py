from feature_extract.input_process.InputOutput import InputOutput
import queue
import numpy as np
import torch

import logging
logger = logging.getLogger('extractor')


class I3DInput(InputOutput):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict, temporalSlide: int,
                 inputLength: int, batchSize: int, inputSize=(224, 224)):
        InputOutput.__init__(self, batch, videoPath, annotations, temporalSlide,
                             inputLength, batchSize, inputSize)

    def prepare_frame(self, frame):
        return frame

    def prepare_input(self, fps):
        if len(self.frames) == self.inputLength:
            anomaly = self.__is_abnormal(fps, 0.7)

            video_clip_np = self.__preprocess_input(self.frames[0:self.inputLength])
            self.frames = self.frames[self.temporalSlide:self.inputLength]

            self.__put_clip(video_clip_np, anomaly)

    def __put_clip(self, video_clip_np, anomaly):

        self.inputClips.append(video_clip_np)
        self.clipNames.append(self.videoName + "_" + str(self.frameCounter).zfill(10))
        self.videoNames.append(self.videoName)
        self.targets.append(anomaly)

    @staticmethod
    def __preprocess_input(clipFrames):
        video_clip_np = np.array(clipFrames, dtype='float32')
        video_clip_np = np.interp(video_clip_np, (video_clip_np.min(), video_clip_np.max()), (-1, +1))
        video_clip_np = np.transpose(video_clip_np, (3, 0, 1, 2)).astype(np.float32)
        return video_clip_np

    def __is_abnormal(self, fps, intersectionThreshold):
        anomaly = 0
        for actionSpace in self.annotations:
            intersectionEnd = min(self.frameCounter, actionSpace["end"] * fps)
            intersectionStart = max(self.frameCounter - self.inputLength, actionSpace["start"] * fps)
            if (intersectionEnd - intersectionStart) / self.inputLength > intersectionThreshold:
                anomaly = 1
                break
            else:
                anomaly = 0
        return anomaly
