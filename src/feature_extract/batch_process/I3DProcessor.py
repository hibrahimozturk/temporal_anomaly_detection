import torch
import queue
import numpy as np

from feature_extract.batch_process.BatchProcessor import BatchProcessor
from feature_extract.models.i3d.i3dpt import I3D

import logging
logger = logging.getLogger('extractor')


class I3DProcessor(BatchProcessor):
    def __init__(self, model_path, batch: queue.Queue,
                 outputs: queue.Queue, dry_run: bool):
        BatchProcessor.__init__(self, batch, outputs, dry_run)
        self.model = I3D(num_classes=400, modality='rgb')
        self.model.eval()
        self.model.load_state_dict(torch.load(model_path))

        if torch.cuda.is_available():
            self.model.cuda()
        logger.info("i3d model has been created")

    def process_batch(self):
        with torch.no_grad():
            if not self.dry_run:
                self.local_batch = torch.from_numpy(np.array(self.local_batch))
                self.local_batch = self.local_batch.cuda()
                output = self.model.feature_extract(self.local_batch)
                output = output.data.cpu().detach().numpy()
            else:
                output = np.zeros((len(self.local_batch), 10))

            return output

    def put_queue(self, output):
        assert output.shape[0] == len(self.clip_names) == len(self.video_names) == len(self.targets), \
            "# of elements are not same"
        for i in range(output.shape[0]):
            self.outputs.put({"out_tensor": output[i],
                              "clip_name": self.clip_names[i],
                              "video_name": self.video_names[i],
                              "anomaly": self.targets[i]})
