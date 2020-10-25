from violance_v2 import ViolenceDetection
from utils.config import Config
from utils.logger import create_logger
import sys

import logging
logger = create_logger("violence")
logger.setLevel(logging.DEBUG)

cfg = Config.fromfile(sys.argv[1])
cfg.mode = "test"
v = ViolenceDetection(cfg)
v.test()
