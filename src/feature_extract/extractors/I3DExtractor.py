from feature_extract.extractors.Extractor import Extractor, EXTRACTORS
from feature_extract.input_process.I3DInput import I3DInput
from feature_extract.batch_process.I3DProcessor import I3DProcessor

import logging
logger = logging.getLogger('extractor')


@EXTRACTORS.register_module(name="i3d")
class I3DExtractor(Extractor):
    def __init__(self, cfg):
        Extractor.__init__(self, cfg)
        logger.info('i3d extractor has been created')

    def get_producer(self, video_path, annotations, **kwargs):
        producer = I3DInput(self.batch, video_path, annotations, kwargs["cfg"].temporal_stride,
                            kwargs["cfg"].input_length,
                            kwargs["cfg"].batch_size, kwargs["cfg"].input_size)
        return producer

    def get_consumer(self, **kwargs):
        consumer = I3DProcessor(kwargs["cfg"].path, self.batch, self.outputs, self.dry_run)
        return consumer
