from feature_extract.extractors.Extractor import Extractor, EXTRACTORS
from feature_extract.input_process.C3DInput import C3DInput
from feature_extract.batch_process.C3DProcessor import C3DProcessor

import logging
logger = logging.getLogger('extractor')


class C3DExtractor(Extractor):
    def __init__(self, cfg):
        Extractor.__init__(self, cfg)

    def get_consumer(self, **kwargs):
        consumer = C3DProcessor(kwargs["cfg"].json, kwargs["cfg"].weight, self.batch, self.outputs, self.dry_run)
        return consumer

    def get_producer(self, video_path, annotations, **kwargs):
        producer = C3DInput(self.batch, video_path, annotations, kwargs["cfg"].temporal_stride,
                            kwargs["cfg"].input_length, kwargs["cfg"].batch_size, kwargs["cfg"].input_mean,
                            kwargs["cfg"].input_size)
        return producer

