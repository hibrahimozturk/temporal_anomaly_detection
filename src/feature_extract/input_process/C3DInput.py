from feature_extract.input_process.InputOutput import InputOutput
import queue
import numpy as np


class C3DInput(InputOutput):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict, temporalSlide: int,
                 inputLength: int, batchSize: int, dataMeanPath, inputSize=(224, 224)):
        InputOutput.__init__(self, batch, videoPath, annotations, temporalSlide,
                             inputLength, batchSize, inputSize)

        self.height_start = 8
        self.width_start = 29

        self.mean_cube = np.load(dataMeanPath)
        self.mean_cube = np.transpose(self.mean_cube, (1, 2, 3, 0))
        self.mean_cube = self.mean_cube[:, self.height_start:self.height_start + self.inputSize[0],
                                        self.width_start:self.width_start + self.inputSize[1], :]

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

    def __preprocess_input(self, clipFrames):
        video_clip_np = np.array(clipFrames, dtype='float32')
        video_clip_np = video_clip_np - self.mean_cube
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

    def prepare_frame(self, frame):
        return frame
