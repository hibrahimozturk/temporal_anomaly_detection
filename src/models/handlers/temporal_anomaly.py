from models.handlers.temporal_action import TemporalActionSegHandler
from abc import ABCMeta, abstractmethod
from models.Loss import TemporalHardPairLoss
from addict import Dict
import torch
import numpy as np

from dataset.ucf_crime import UCFCrimeTemporal, collate_fn_precomp
from torch.utils.data import DataLoader


class TemporalAnomalyDetectionHandler(TemporalActionSegHandler):
    def __init__(self, cfg):
        __metaclass__ = ABCMeta
        TemporalActionSegHandler.__init__(self, cfg)
        self.num_classes = 2
        self.thresholds = cfg.evaluation.thresholds

    @abstractmethod
    def get_model(self, model_cfg):
        pass

    @abstractmethod
    def model_forward(self, data, evaluate=False):
        pass

    def get_dataloaders(self, data_cfg, mode):

        if mode == "train":
            train_dataset = UCFCrimeTemporal(data_cfg, split="train")

            train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                      num_workers=data_cfg.num_workers,
                                      collate_fn=collate_fn_precomp,
                                      pin_memory=True)

            val_dataset = UCFCrimeTemporal(data_cfg, split="val")

            val_loader = DataLoader(val_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    num_workers=data_cfg.num_workers,
                                    collate_fn=collate_fn_precomp)

            return train_loader, val_loader

        elif mode == "test":
            test_dataset = UCFCrimeTemporal(data_cfg, split="test")

            test_loader = DataLoader(test_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                     num_workers=data_cfg.num_workers,
                                     collate_fn=collate_fn_precomp)
            return test_loader

    def filter_data(self, data):
        mask = data["anomalies"] != self.mask_value
        for key in data:
            data[key] = data[key][mask]
        return data

    def clip_outputs(self, output, clip_targets, clip_names):
        clip_data = Dict()
        clip_data.anomalies = clip_targets.view(-1).detach().cpu().numpy()
        clip_data.clip_names = np.array(clip_names).reshape(-1)
        clip_data.outputs = output.view(-1).detach().cpu().numpy()

        clip_data = self.filter_data(clip_data)
        clip_data.clip_names = clip_data.clip_names.tolist()

        return clip_data

    def visualize_outputs(self, video_clips, exp_dir):
        pass

    @staticmethod
    def append_overlapped_clips(clip_data):
        clip_dicts = Dict(predictions=dict(),
                          targets=dict())

        for clip_name, prediction, target in zip(clip_data.clip_names, clip_data.outputs,
                                                 clip_data.anomalies):
            if clip_name not in clip_dicts.predictions:
                clip_dicts.predictions[clip_name] = []
            if clip_name not in clip_dicts.targets:
                clip_dicts.targets[clip_name] = []
            clip_dicts.predictions[clip_name].append(prediction)
            clip_dicts.targets[clip_name].append(target)
        return clip_dicts

    @staticmethod
    def create_losses(loss_cfg):
        losses = Dict()
        for loss_dict in loss_cfg:
            if loss_dict.type == "mse":
                losses["mse"] = Dict(loss=torch.nn.MSELoss(),
                                     factor=loss_dict.factor)
            elif loss_dict.type == "thp":
                losses["thp"] = Dict(loss=TemporalHardPairLoss(margin=loss_dict.params.margin,
                                                               max_violation=loss_dict.params.max_violation,
                                                               measure=loss_dict.params.measure),
                                     factor=loss_dict.factor)
            else:
                raise Exception("loss not handled")

            # TODO: cross entropy loss

        return losses

    @staticmethod
    def move_2_gpu(data):
        input_data, masks, anomalies, category, _, clip_names = data
        gpu_data = Dict()
        if torch.cuda.is_available():
            gpu_data.input = input_data.float().cuda()
            gpu_data.anomalies = anomalies.float().cuda()
            gpu_data.masks = masks.float().cuda()
        gpu_data.clip_names = clip_names
        return gpu_data

    def temporal_score(self, iou_list, video_clips, bg_class=0):
        threshold_scores = Dict()
        for threshold in self.thresholds:
            for video_name in video_clips:
                video_clips[video_name].predictions = (np.array(video_clips[video_name].predictions) > threshold).tolist()
            output_scores = super().temporal_score(iou_list, video_clips, bg_class=0)
            threshold_scores["thr_{:.2f}".format(threshold)] = output_scores
        return threshold_scores

    def calculate_loss(self, output, target):
        total_loss = 0
        loss_outputs = dict()
        for loss_type, loss_cfg in self.losses.items():
            if loss_type == "mse":
                partial_loss = self.mse_loss_calculate(output, target)
            elif loss_type == "thp":
                partial_loss = self.thp_loss_calculate(output, target)
            else:
                raise Exception("loss not handled")

            loss_outputs[loss_type] = partial_loss

            factor = torch.tensor(float(loss_cfg.factor), requires_grad=True).float()
            if torch.cuda.is_available():
                factor = factor.cuda()
            total_loss += factor * partial_loss
        return total_loss, loss_outputs

    def thp_loss_calculate(self, output, target):
        mask = (target.view(-1) != self.mask_value).nonzero().squeeze()
        hp_values = torch.zeros_like(target.view(-1))
        hp_values[mask] = self.losses.thp.loss(target.view(-1)[mask], output.view(-1)[mask])
        hp_values = hp_values.reshape((target.shape[0], target.shape[1]))

        thp_loss = 0
        for target_win, output_win, hp_value_win in zip(target, output, hp_values):
            clip_filter = (target_win != self.mask_value).nonzero().squeeze()
            thp_loss += hp_value_win[clip_filter].mean()

        return thp_loss

    def mse_loss_calculate(self, output, target):
        mse_loss = 0
        for target_win, output_win in zip(target, output):
            clip_filter = (target_win != self.mask_value).nonzero().squeeze()
            mse_loss += self.losses.mse.loss(output_win[clip_filter], target_win[clip_filter])
        return mse_loss
