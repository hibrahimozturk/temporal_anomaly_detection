from abc import ABCMeta, abstractmethod
import torch
import numpy as np
from utils.registry import Registry
from addict import Dict

import logging

logger = logging.getLogger("violence")

HANDLERS = Registry("handlers")


class ModelHandler:
    def __init__(self, cfg):
        __metaclass__ = ABCMeta
        self.mode = cfg.mode
        logger.info("model handler switched to {} mode".format(self.mode))

        self.model = self.get_model(cfg.model)
        self.losses = self.create_losses(cfg.model.losses)

        if self.mode == "train":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.train.optimizer.lr,
                                              betas=(0.5, 0.9), eps=1e-08,
                                              weight_decay=0, amsgrad=False)
            if hasattr(cfg.train.optimizer, "scheduler"):
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                 step_size=cfg.train.optimizer.scheduler.step_size,
                                                                 gamma=cfg.train.optimizer.scheduler.gamma)
            self.train_loader, self.val_loader = self.get_dataloaders(cfg.dataset, mode=self.mode)

        elif self.mode == "test":
            self.test_loader = self.get_dataloaders(cfg.dataset, mode=self.mode)

        self.model_cast_move()

    @abstractmethod
    def get_model(self, model_cfg):
        pass

    @abstractmethod
    def get_dataloaders(self, data_cfg):
        pass

    @abstractmethod
    def model_forward(self, data, evaluate=False):
        pass

    @abstractmethod
    def calculate_score(self, epoch_dict):
        pass

    @abstractmethod
    def visualize_outputs(self, output_data, exp_dir):
        pass

    @staticmethod
    def init_epoch_dict():
        return Dict(losses=dict())

    def init_train_epoch_dict(self):
        epoch_dict = self.init_epoch_dict()
        return epoch_dict

    @abstractmethod
    def init_eval_epoch_dict(self):
        pass

    @staticmethod
    @abstractmethod
    def create_losses(loss_cfg):
        pass

    def train_iteration(self, data, epoch_dict):
        self.optimizer.zero_grad()
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        loss, report, _, loss_dict = self.model_forward(data)

        for loss_key in loss_dict:
            if loss_key not in epoch_dict.losses:
                epoch_dict.losses[loss_key] = []
            epoch_dict.losses[loss_key].append(loss_dict[loss_key])

        report["lr"] = self.optimizer.param_groups[0]["lr"]
        logger.debug("iteration report:\n{}".format(report))
        loss.backward()
        self.optimizer.step()
        return report, epoch_dict

    def eval_iteration(self, data, epoch_dict):
        loss, report, clip_dicts, loss_dict = self.model_forward(data, evaluate=True)

        for loss_key in loss_dict:
            if loss_key not in epoch_dict.losses:
                epoch_dict.losses[loss_key] = []
            epoch_dict.losses[loss_key].append(loss_dict[loss_key])

        epoch_dict.predictions.update(clip_dicts.predictions)
        epoch_dict.targets.update(clip_dicts.targets)
        return report, epoch_dict

    @staticmethod
    def iter_info(report_data):
        info = "Loss {:.3f}".format(report_data["total_loss"])
        return info

    @staticmethod
    def epoch_report(epoch_dict):
        report = dict()
        report['message'] = ""
        for loss_key in epoch_dict.losses:
            report["epoch/{}".format(loss_key)] = float(np.array(epoch_dict.losses[loss_key]).mean())
            report['message'] += " {}: {:.2f} ".format(loss_key, report["epoch/{}".format(loss_key)])
        logger.debug("\n{}".format(report))
        return report

    def save_variables(self):
        variables_dict = Dict(optimizer=self.optimizer.state_dict(),
                              model=self.model.state_dict())
        if hasattr(self, "scheduler"):
            variables_dict.scheduler = self.scheduler.state_dict()

        return variables_dict

    def load_variables(self, variables):
        self.model.load_state_dict(variables["model"])
        if self.mode == "train":
            self.optimizer.load_state_dict(variables["optimizer"])
            if hasattr(self, "scheduler"):
                self.scheduler.load_state_dict(variables["scheduler"])
        self.model_cast_move()

    def model_cast_move(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("model has been moved to gpu")
        self.model = self.model.float()
