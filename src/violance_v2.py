import os
import sys
import shutil

import torch
from addict import Dict
from torch.utils.tensorboard import SummaryWriter

from models.handlers.handler import HANDLERS
from utils.utils import ignore_func
from utils.config import Config
from utils.logger import create_logger, logging
from utils.utils import input_with_timeout

logger = create_logger("violence")
logger.setLevel(logging.INFO)


class ViolenceDetection:
    def __init__(self, cfg):
        logger.info(cfg.pretty_text)
        self.handler = HANDLERS.get(cfg.model_type)(cfg)

        if hasattr(cfg, "load_from"):
            self.load_checkpoint(cfg.load_from)

        self.epoch = -1
        self.step_counter = 0
        self.config = Dict(log_step=cfg.log_step)

        if hasattr(cfg, "train"):
            self.config.exp_dir = cfg.train.exp_dir
            self.config.chk_dir = os.path.join(cfg.train.exp_dir, "checkpoints")

            if not os.path.exists(self.config.chk_dir):
                os.makedirs(self.config.chk_dir)

            self.backup_src()
            self.num_epochs = cfg.train.num_epochs

            if hasattr(cfg.train, "tensorboard") and getattr(cfg.train, "tensorboard"):
                self.writer = SummaryWriter(log_dir=cfg.train.exp_dir)

    def train(self):
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_dict = self.handler.init_train_epoch_dict()
            self.handler.model.train()
            for step, data in enumerate(self.handler.train_loader):
                iter_report, epoch_dict = self.handler.train_iteration(data, epoch_dict)
                self.step_counter += 1
                info = self.iter_info("train", step, len(self.handler.train_loader), iter_report)
                if step % self.config.log_step == 0:
                    print(info)
                self.report(iter_report, "train")
            epoch_report = self.handler.epoch_report(epoch_dict)
            self.report(epoch_report, "train/epoch")
            self.save_checkpoint(os.path.join(self.config.chk_dir, "epoch-{}.pth".format(self.epoch)))
            self.eval_epoch()

    def test(self):
        self.eval_epoch(split="test")

    def eval_epoch(self, split="val"):
        self.handler.model.eval()
        epoch_dict = self.handler.init_eval_epoch_dict()
        dataloader = getattr(self.handler, "{}_loader".format(split))
        for step, data in enumerate(dataloader):
            iter_report, epoch_dict = self.handler.eval_iteration(data, epoch_dict)
            info = self.iter_info(split, step, len(dataloader), iter_report)
            if step % self.config.log_step == 0:
                print(info)
        epoch_report = self.handler.epoch_report(epoch_dict)
        epoch_report, output_data = self.handler.calculate_score(epoch_dict, epoch_report)
        self.report(epoch_report, split)
        # TODO: visualize outputs

    def report(self, report_data, phase):
        if 'message' in report_data:
            print('-'*50)
            print("{:5} [{:3}] : {}".format(phase, self.epoch, report_data['message']))
            print('-'*50)
            del report_data['message']
        if hasattr(self, "writer"):
            for label, data in report_data.items():
                absolute_label = phase + "/" + str(label)
                if type(data) == dict or type(data) == Dict:
                    self.report(data, absolute_label)
                else:
                    self.writer.add_scalar(absolute_label, data, self.step_counter)
        else:
            logger.debug("report has not written anywhere because tensorboard was not defined")

    def iter_info(self, phase, step, total_step, report_data):
        pre_info = "{:5} [{:3}] : [{:3} / {}]   global_step:{:5} ".format(phase, self.epoch, step,
                                                                          total_step, self.step_counter)
        info = self.handler.iter_info(report_data)
        return pre_info + info

    def save_checkpoint(self, path):
        save_dict = self.handler.save_variables()
        save_dict.step_counter = self.step_counter
        torch.save(save_dict.to_dict(), path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.step_counter = checkpoint["step_counter"]
        self.handler.load_variables(checkpoint)

    def backup_src(self):
        src_folder = os.path.abspath(__file__).rsplit("/", 1)[0]
        dst_folder = os.path.join(self.config.exp_dir, "backup_code")

        if os.path.exists(dst_folder):
            answer = input_with_timeout("Do you want to delete already exist backup code / keep old backup?(y/n): ",
                                        5, default_answer='y')
            if answer == "y":
                shutil.rmtree(dst_folder)
            else:
                return

        shutil.copytree(src_folder, dst_folder, ignore=ignore_func)


if __name__ == "__main__":
    cfg = Config.fromfile(sys.argv[1])
    cfg.mode = "train"
    v = ViolenceDetection(cfg)
    v.train()
