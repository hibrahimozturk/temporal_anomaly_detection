from models.handlers.handler import ModelHandler
from abc import ABCMeta, abstractmethod
from addict import Dict
import numpy as np
import os
from metrics import f_score, calc_f1
from utils.utils import visualize_temporal_action


class TemporalActionSegHandler(ModelHandler):
    def __init__(self, cfg):
        __metaclass__ = ABCMeta
        ModelHandler.__init__(self, cfg)
        self.mask_value = cfg.dataset.mask_value
        self.num_classes = 1
        self.iou_list = cfg.evaluation.iou_list

    @abstractmethod
    def get_dataloaders(self, data_cfg):
        pass

    @abstractmethod
    def get_model(self, model_cfg):
        pass

    @abstractmethod
    def model_forward(self, data, evaluate=False):
        pass

    @staticmethod
    @abstractmethod
    def create_losses(loss_cfg):
        pass

    def init_eval_epoch_dict(self):
        epoch_dict = self.init_epoch_dict()
        val_epoch_dict = Dict(targets=dict(),
                              predictions=dict())
        epoch_dict.update(val_epoch_dict)
        return epoch_dict

    def calculate_score(self, epoch_dict, epoch_report):
        video_clips = self.organize_video_clip(epoch_dict)
        temporal_scores = self.temporal_score(iou_list=self.iou_list,
                                              video_clips=video_clips)
        epoch_report['scores'] = temporal_scores
        epoch_report['message'] += self.score_message(temporal_scores)

        return epoch_report, video_clips

    def visualize_outputs(self, video_clips, exp_dir):
        output_dir = os.path.join(exp_dir, "output")
        if os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.temporal_visualize(video_clips, output_dir)

    @staticmethod
    def temporal_visualize(video_clips, output_dir):
        for video_name, clips in video_clips.items():
            save_path = os.path.join(output_dir, video_name.split(".")[0] + ".png")
            visualize_temporal_action(clips.predictions, clips.targets, save_path, video_name)

    @staticmethod
    def score_message(scores):
        message = "\n"
        for thresh, iou_scores in scores.items():
            for score_set in iou_scores:
                message += "#" * 10 + " " + score_set + " " + "#" * 10 + "\n"
                for iou_thresh, thresh_scores in iou_scores[score_set].items():
                    message += "{} {} - f1: {:.2f} pr: {:.2f} rc: {:.2f}\n".format(thresh, iou_thresh,
                                                                                   thresh_scores['f1'],
                                                                                   thresh_scores['precision'] * 100,
                                                                                   thresh_scores['recall'] * 100)
        return message

    def organize_video_clip(self, epoch_dict):
        video_clips = dict()

        dataloader = getattr(self, "{}_loader".format(self.mode))
        video_clip_list = dataloader.dataset.__getVideoClips__()
        for video_name, clip_list in video_clip_list.items():
            clips = Dict(predictions=[], targets=[])
            for clip_name in clip_list:
                clips.predictions.append(np.mean(np.array(epoch_dict.predictions[clip_name])))
                clips.targets.append(np.mean(np.array(epoch_dict.targets[clip_name])))
            video_clips[video_name] = clips
        return video_clips

    def temporal_score(self, iou_list, video_clips, bg_class=0):
        output_scores = self.__init_out_score_dict(iou_list)

        for iou in iou_list:
            confusion_mat = Dict(fp=0, tp=0, fn=0)
            class_confusion_mat = Dict()
            for c in range(self.num_classes):
                class_confusion_mat[c] = Dict(fp=0, tp=0, fn=0)
            for video_name, clip_list in video_clips.items():
                clips = video_clips[video_name]
                for c in range(self.num_classes):
                    targets = (np.array(clips.targets) == c)
                    predictions = (np.array(clips.predictions) == c)
                    tp1, fp1, fn1 = f_score(predictions, targets, iou, bg_class=0)

                    class_confusion_mat[c].fp += fp1
                    class_confusion_mat[c].tp += tp1
                    class_confusion_mat[c].fn += fn1

                tp1, fp1, fn1 = f_score(clips.predictions, clips.targets, iou, bg_class=bg_class)

                confusion_mat.tp += tp1
                confusion_mat.fp += fp1
                confusion_mat.fn += fn1

            for c in range(self.num_classes):
                output_scores["class_{}".format(c)]["iou_{:.2f}".format(iou)] = calc_f1(class_confusion_mat[c].fn,
                                                                                        class_confusion_mat[c].fp,
                                                                                        class_confusion_mat[c].tp)

            output_scores.overall["iou_{:.2f}".format(iou)] = calc_f1(confusion_mat.fn,
                                                                      confusion_mat.fp,
                                                                      confusion_mat.tp)

        return output_scores

    def __init_out_score_dict(self, iou_list):
        output_scores = Dict(overall=dict())
        for c in range(self.num_classes):
            output_scores["class_{}".format(c)] = dict()
            for iou in iou_list:
                output_scores["class_{}".format(c)]["iou_{:.2f}".format(iou)] = Dict(f1=0, precesion=0, recall=0)
                output_scores.overall["iou_{:.2f}".format(iou)] = Dict(f1=0, precesion=0, recall=0)
        return output_scores
