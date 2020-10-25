import sys
from feature_extract.extractors.Extractor import EXTRACTORS
from feature_extract.utils.config import Config
from feature_extract.utils.logger import create_logger
import logging

logger = create_logger("extractor")
logger.setLevel(logging.DEBUG)

cfg = Config.fromfile(sys.argv[1])
ext = EXTRACTORS.get(cfg.extractor_type)(cfg)
ext()
