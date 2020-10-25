from models.handlers.handler import HANDLERS
from models.handlers.temporal_anomaly import TemporalAnomalyDetectionHandler

from models.TCN import EDTCN
import torch
import numpy as np
from addict import Dict


@HANDLERS.register_module(name="edtcn")
class EDCTNHandler(TemporalAnomalyDetectionHandler):
    def __init__(self, cfg):
        TemporalAnomalyDetectionHandler.__init__(self, cfg)

    def get_model(self, model_cfg):
        model = EDTCN(featureSize=model_cfg.feature_size, kernelSize=model_cfg.kernel_size)
        return model

    def model_forward(self, data, evaluate=False):
        data = self.move_2_gpu(data)

        report = dict()
        clip_dicts = None
        loss_dict = {"{}_loss".format(loss_type): 0 for loss_type, loss_cfg in self.losses.items()}

        outputs = self.model(data["input"])

        loss, loss_parts = self.calculate_loss(outputs, data['anomalies'])
        for loss_type, loss_value in loss_parts.items():
            loss_dict["{}_loss".format(loss_type)] = loss_value.item()
            report["loss/{}".format(loss_type)] = loss_value.item()

        loss_dict['total_loss'] = loss.item()
        report['total_loss'] = loss.item()

        if evaluate:
            c_data = self.clip_outputs(outputs, data['anomalies'], data['clip_names'])
            clip_dicts = self.append_overlapped_clips(c_data)

        return loss, report, clip_dicts, loss_dict
