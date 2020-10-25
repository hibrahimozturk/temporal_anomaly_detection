from abc import ABCMeta, abstractmethod
import queue
import threading
import time

import logging
logger = logging.getLogger('extractor')


class BatchProcessor(threading.Thread):
    def __init__(self, batch: queue.Queue,
                 outputs: queue.Queue,
                 dry_run: bool):
        __metaclass__ = ABCMeta
        threading.Thread.__init__(self)

        self.batch = batch
        self.outputs = outputs
        self.dry_run = dry_run

        self.local_batch = []
        self.clip_names = []
        self.targets = []
        self.video_names = []

        return

    def run(self):
        while True:
            if self.batch.qsize() > 0:
                element = self.batch.get()
                if element is None:
                    self.__kill_thread()
                    break
                else:
                    self.__extract_element(element)
            else:
                time.sleep(1)

            if len(self.local_batch) != 0:
                output = self.process_batch()
                assert output is not None, "Batch not processed properly"
                self.put_queue(output)
                self.__clean_elements()

        return 0

    @abstractmethod
    def process_batch(self):
        pass

    @abstractmethod
    def put_queue(self, output):
        pass

    def __clean_elements(self):
        self.local_batch = []
        self.clip_names =[]
        self.video_names = []
        self.targets = []

    def __extract_element(self, element):
        self.local_batch = element["inputClip"]
        self.clip_names = element["clipName"]
        self.video_names = element["videoName"]
        self.targets = element["target"]
        return

    def __kill_thread(self):
        if not self.batch.empty():
            logger.debug("thread will be killed but batch is not empty")
        while not self.outputs.empty():
            time.sleep(2)
        self.outputs.put(None)
        logger.info("thread has been killed")
