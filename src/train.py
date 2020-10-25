from violance import ViolenceDetection
from utils.logger import create_logger
from utils.config import Config
import sys

import logging
logger = create_logger("violence")
logger.setLevel(logging.DEBUG)

cfg = Config.fromfile(sys.argv[1])
vlcdtr = ViolenceDetection(cfg, mode="train")
vlcdtr.train()
