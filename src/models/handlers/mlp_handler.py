from models.handlers.handler import HANDLERS
from models.handlers.temporal_anomaly import TemporalAnomalyDetectionHandler

from dataset.ucf_crime import UCFCrimeDataset
from torch.utils.data import DataLoader
from models.MLP import MLP


@HANDLERS.register_module(name="mlp")
class MLPHandler(TemporalAnomalyDetectionHandler):
    def __init__(self, cfg):
        TemporalAnomalyDetectionHandler.__init__(self, cfg)

    def get_dataloaders(self, data_cfg, mode):
        if mode == "train":
            train_dataset = UCFCrimeDataset(data_cfg, split="train")

            train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,
                                      num_workers=data_cfg.num_workers)

            val_dataset = UCFCrimeDataset(data_cfg, split="val")

            val_loader = DataLoader(val_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                    num_workers=data_cfg.num_workers)

            return train_loader, val_loader

        elif mode == "test":
            test_dataset = UCFCrimeDataset(data_cfg, split="test")

            test_loader = DataLoader(test_dataset, batch_size=data_cfg.batch_size, shuffle=False,
                                     num_workers=data_cfg.num_workers)

            return test_loader

    def get_model(self, model_cfg):
        model = MLP(featureSize=model_cfg.feature_size)
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

    def filter_data(self, data):
        return data

    def mse_loss_calculate(self, output, target):
        return self.losses.mse.loss(output, target)

    def thp_loss_calculate(self, output, target):
        raise Exception("loss not handled")
