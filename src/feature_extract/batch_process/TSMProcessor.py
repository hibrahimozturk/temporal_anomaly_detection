from feature_extract.batch_process.BatchProcessor import BatchProcessor
from feature_extract.models.tsm.feature_extractor import TSMFeatureExtractor

import torch
import numpy as np
import queue

import logging
logger = logging.getLogger('extractor')


class TSMProcessor(BatchProcessor):
    def __init__(self, model_path, input_length, batch: queue.Queue, outputs: queue.Queue,
                 dry_run: bool, num_crops=1):
        BatchProcessor.__init__(self, batch, outputs, dry_run)
        self.model = TSMFeatureExtractor(model_path, segments=input_length, crops=num_crops)
        logger.info("tsm model has been created")

    def process_batch(self):
        with torch.no_grad():
            if not self.dry_run:
                out_tensor = self.model.feature_extract(self.local_batch)
                out_tensor = out_tensor.data.cpu().detach().numpy()
            else:
                out_tensor = np.zeros((len(self.local_batch), len(self.clip_names[0]), 10))

        return out_tensor

    def put_queue(self, output):
        assert output.shape[0] == len(self.clip_names) == len(self.video_names) == len(self.targets), \
            "[TSM Processor] # of elements are not same"
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                self.outputs.put({"out_tensor": output[i][j],
                                  "clip_name": self.clip_names[i][j],
                                  "video_name": self.video_names[i][j],
                                  "anomaly": self.targets[i][j]})
