import os
import time

import queue
import json
from feature_extract.output_writer.Writer import Writer
from abc import ABCMeta, abstractmethod
from feature_extract.utils.registry import Registry

import logging
logger = logging.getLogger('extractor')
EXTRACTORS = Registry("extractors")


class Extractor:
    def __init__(self, cfg):
        __metaclass__ = ABCMeta

        logger.info(cfg.pretty_text)

        self.num_producers = cfg.extractor.num_producers
        self.dry_run = cfg.extractor.dry_run
        self.top_k = cfg.extractor.top_k if hasattr(cfg.extractor, "top_k") else None

        self.batch = queue.Queue(10)
        self.outputs = queue.Queue()

        self.producers = []

        self.writer = Writer(self.outputs, cfg.output_writer.clip_folder,
                             cfg.output_writer.json_path, cfg.extractor.categories,
                             self.dry_run)
        self.consumer = self.get_consumer(cfg=cfg.model)
        self.producer_cfg = cfg.input_processor

        self.video_folder = cfg.extractor.video_folder
        with open(cfg.extractor.temporal_annotions) as fp:
            self.temporal_annotations = json.load(fp)

    @abstractmethod
    def get_consumer(self, **kwargs):
        pass

    @abstractmethod
    def get_producer(self, video_path, annotations, **kwargs):
        pass

    def __call__(self):
        logger.debug("extraction process starts")
        self.consumer.start()
        self.writer.start()
        for indx, (video_name, annotations) in enumerate(self.temporal_annotations.items()):
            if indx == self.top_k:
                logger.info("top {} videos has been processed".format(self.top_k))
                break
            video_path = os.path.join(self.video_folder, video_name)
            producer = self.get_producer(video_path, annotations, cfg=self.producer_cfg)

            producer.start()
            time.sleep(1)
            self.producers.append(producer)
            self.__wait_producers()
        self.__finalize()
        logger.debug("extraction process has been finished")

    def __wait_producers(self):
        while len(self.producers) == self.num_producers:
            tempList = []
            for producer in self.producers:
                if producer.is_alive() is True:
                    tempList.append(producer)
            self.producers = tempList
            logger.debug("# producer threads {}".format(len(self.producers)))
            time.sleep(1)

    def __finalize(self):
        for producer in self.producers:
            producer.join()

        logger.debug("finalize signal to batch queue")
        time.sleep(5)
        self.batch.put(None)
        self.consumer.join()
        self.writer.join()

