import queue
import cv2
import torch
import time
import threading
import numpy as np
from abc import ABCMeta, abstractmethod

import logging
logger = logging.getLogger('extractor')


class InputOutput(threading.Thread):
    def __init__(self, batch: queue.Queue, videoPath: str, annotations: dict, temporalSlide: int,
                 inputLength: int, batchSize: int, inputSize=(224, 224)):
        __metaclass__ = ABCMeta
        threading.Thread.__init__(self)

        self.batch = batch
        self.videoPath = videoPath
        self.videoName = self.videoPath.split("/")[-1].split(".")[0]
        self.annotations = annotations
        self.temporalSlide = temporalSlide
        self.inputLength = inputLength
        self.batchSize = batchSize
        self.inputSize = inputSize

        self.frames = []
        self.inputClips = []
        self.clipNames = []
        self.targets = []
        self.videoNames = []

        self.frameCounter = 0
        self.clipFrame = 0
        logger.info("clips of {} are extracting".format(videoPath.split("/")[-1]))

    def run(self):
        capture = cv2.VideoCapture(self.videoPath)
        fps = capture.get(cv2.CAP_PROP_FPS)

        while True:
            ret, img = capture.read()
            if not ret:
                if len(self.inputClips) != 0:
                    self.__put_queue()
                logger.info("{} has been finished".format(self.videoPath.split("/")[-1]))
                return 0
            img = cv2.resize(img, self.inputSize)
            img = self.prepare_frame(img)

            self.frames.append(img)
            self.frameCounter += 1

            self.prepare_input(fps)

            self.__queue_full()
            if len(self.inputClips) == self.batchSize:
                logger.debug("targets: {}".format(self.targets))
                self.__put_queue()
                logger.debug("batch size: {} (new batch)".format(self.batch.qsize()))
            # TODO: last clips are lost, solve

        return 0

    def __put_queue(self):
        assert len(self.inputClips) == len(self.clipNames) == len(self.videoNames) == len(self.targets), \
            "# of elements are not same"
        self.inputClips = self.inputClips
        self.batch.put({"inputClip": self.inputClips,
                        "clipName": self.clipNames,
                        "videoName": self.videoNames,
                        "target": self.targets,
                        "batchSize": len(self.inputClips)})
        self.inputClips, self.clipNames, self.targets, self.videoNames = [], [], [], []

    @abstractmethod
    def prepare_input(self, fps):
        pass

    @abstractmethod
    def prepare_frame(self, frame):
        pass

    def __queue_full(self):
        while self.batch.full():
            logger.debug("batch size: {} (full)".format(self.batch.qsize()))
            time.sleep(2)
