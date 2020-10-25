import torch
from torch.utils.tensorboard import SummaryWriter
from param import args
from dataset.ucf_crime import get_dataloaders
from models.MLP import MLP
from models.TCN import EDTCN
from models.MSTCN import MultiStageModel
import sklearn.metrics as sklrn
import os
import shutil
from metrics import f_score
import numpy as np
from tqdm.autonotebook import tqdm
from utils import utils
from models.Loss import TemporalHardPairLoss
from metrics import calc_f1

# from models.MCBTCN import MultiClassBinaryTCN


class ViolenceDetection:
    def __init__(self, cfg):
        trainLoader, valLoader = get_dataloaders(args.trainNormalFolder, args.trainNormalAnnotations,
                                                 args.trainAbnormalFolder, args.trainAbnormalAnnotations,
                                                 args.trainNormalTopK,
                                                 args.valNormalFolder, args.valNormalAnnotations,
                                                 args.valAbnormalFolder, args.valAbnormalAnnotations,
                                                 args.valNormalTopK,
                                                 args.batchSize, args.numWorkers, args.model,
                                                 args.windowSize, args.subWindows, args.featureSize,
                                                 args.maxVideoSize)

        self.modelType = args.model
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.expFolder = args.expFolder
        self.maskValue = args.maskValue
        self.stepCounter = 0
        self.bestAUC = 0
        self.noNormalSegmentation = args.noNormalSegmentation
        self.lossType = args.loss

        if args.model == "mlp":
            self.model = MLP(featureSize=args.featureSize)
        elif args.model == "tcn":
            self.model = EDTCN(featureSize=args.featureSize, kernelSize=args.kernelSize)
        elif args.model == "mstcn":
            self.model = MultiStageModel(num_stages=args.numStages, num_layers=args.numLayers,
                                         num_f_maps=args.numFeatureMaps,
                                         dim=args.featureSize, ssRepeat=args.firstStageRepeat)
            print("[Info] MS-TCN W{}-S{}-L{} have been created".format(args.windowSize, args.numStages, args.numLayers))
        # elif args.model == "mcbtcn":
        #     self.model = MultiClassBinaryTCN(numClassStages=args.numClassStages, numBinaryStages=args.numBinaryStages,
        #                                      num_layers=args.numLayers, num_f_maps=args.numFeatureMaps,
        #                                      dim=args.featureSize, numClasses=16)

        self.model = self.model.float()

        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learningRate,
                                              betas=(0.5, 0.9), eps=1e-08, weight_decay=0, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.schedulerStepSize,
                                                         gamma=args.schedulerGamma)
        if args.modelPath:
            self.loadCheckpoint(args.modelPath)
            print("[Info] Model have been loaded at {}".format(args.modelPath))

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model = self.model.float()

        self.ceLoss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.ASLoss = TemporalHardPairLoss(max_violation=True, margin=args.adLossMargin, measure="output")
        self.mseLoss = torch.nn.MSELoss()
        self.lossLambda = args.adLossLambda

        self.writer = None
        if not args.test:
            self.writer = SummaryWriter(log_dir=args.expFolder)

    def train(self):
        # currentScore = self.binaryValidation(-1)
        for epoch in range(args.epoch):
            # print("########## {} Epoch Training Starts ##########".format(epoch))
            progress_bar = tqdm(self.trainLoader)
            for step, (data, masks, anomaly, category, _, _) in enumerate(progress_bar):
                self.model.train()
                # self.scheduler.step()
                self.optimizer.zero_grad()
                self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]["lr"], self.stepCounter)

                # if self.modelType == "tcn" or self.modelType == "mstcn":
                #     anomaly = anomaly.view(-1)

                if torch.cuda.is_available():
                    data = data.float().cuda()
                    anomaly = anomaly.float().cuda()
                    masks = masks.float().cuda()
                    category = category.long().cuda()

                if self.modelType == "mstcn":
                    outputs = self.model(data, masks)
                    loss = 0

                    for i, output in enumerate(outputs):
                        mseLoss = 0
                        adLoss = 0
                        # output = output.view(-1)
                        output = output.squeeze()
                        mask = (anomaly.view(-1) != self.maskValue).nonzero().squeeze()
                        adValues = torch.zeros_like(anomaly.view(-1))
                        if self.lossType == "ad":
                            adValues[mask] = self.ASLoss(anomaly.view(-1)[mask], output.view(-1)[mask])
                            adValues = adValues.reshape((anomaly.shape[0], anomaly.shape[1]))
                            adValues = adValues.transpose(1, 0)

                        anomalies = anomaly.transpose(1, 0)
                        output = output.transpose(1, 0)
                        # calculate each window loss separately
                        for anomalyT, outputT, adValueT, maskT in zip(anomalies, output, adValues, masks):
                            maskIndex = (anomalyT != self.maskValue).nonzero().squeeze()
                            mseLoss += self.mseLoss(outputT[maskIndex], anomalyT[maskIndex])
                            if self.lossType == "ad":
                                adLoss += adValueT[maskIndex].mean()
                        self.writer.add_scalar("Loss/Layer-MSE-{}".format(i), mseLoss.item(), self.stepCounter)
                        loss += mseLoss
                        if self.lossType == "ad":
                            loss += torch.tensor(float(self.lossLambda), requires_grad=True).float().cuda() * adLoss
                            self.writer.add_scalar("Loss/Layer-AD-{}".format(i), adLoss.item(), self.stepCounter)

                elif self.modelType == "mcbtcn":
                    classOutputs, binaryOutputs = self.model(data, masks)
                    loss = 0

                    for i, output in enumerate(classOutputs):
                        loss += self.ceLoss(output, category)

                    for i, output in enumerate(binaryOutputs):
                        # eliminate end of the windows
                        mask = (anomaly.view(-1) != self.maskValue).nonzero()
                        filteredOutput = output.mean(1).view(-1)[mask]
                        filteredAnomaly = anomaly.view(-1)[mask]
                        loss += self.mseLoss(filteredOutput, filteredAnomaly)
                elif self.modelType == "tcn":
                    anomaly = anomaly.view(-1)
                    outputs = self.model(data)
                    outputs = outputs[(anomaly != self.maskValue).nonzero().squeeze()]
                    anomaly = anomaly[(anomaly != self.maskValue).nonzero().squeeze()]
                    loss = self.mseLoss(outputs.squeeze(), anomaly)
                else:
                    outputs = self.model(data)
                    outputs = outputs[(anomaly != self.maskValue).nonzero().squeeze()]
                    anomaly = anomaly[(anomaly != self.maskValue).nonzero().squeeze()]
                    loss = self.mseLoss(outputs.squeeze(), anomaly)

                loss.backward()

                if self.stepCounter % 10 == 0:
                    self.writer.add_scalar("Loss/Train", loss.item(), self.stepCounter)
                    progress_bar.set_description(
                        "Train [{}]:[{}/{}] Global Step:{} Loss: {:.2f}".format(epoch, step, self.trainLoader.__len__(),
                                                                                self.stepCounter, loss.item()))
                self.optimizer.step()
                self.stepCounter += 1

            currentScore = self.binaryValidation(epoch)

            self.saveCheckpoint(os.path.join(self.expFolder, "last.pth"))
            if currentScore > self.bestAUC:
                self.bestAUC = currentScore
                shutil.copy(os.path.join(self.expFolder, "last.pth"),
                            os.path.join(self.expFolder, "best.pth"))

        currentScore = self.binaryValidation(-1)

    def binaryValidation(self, epoch):
        self.model.eval()
        anomalyPredictions = []
        anomalyTargets = []
        predictions = {}
        targets = {}

        thresholds = [0.5, 0.75, 0.9]
        IOUs = [0.1, 0.25, 0.5]
        score = 0

        with torch.no_grad():
            progress_bar = tqdm(self.valLoader)
            for step, (data, masks, anomaly, category, _, clipNames) in enumerate(progress_bar):

                if self.modelType == "tcn" or self.modelType == "mstcn" or self.modelType == "mcbtcn":
                    anomaly = anomaly.view(-1)
                    clipNames = np.array(clipNames).reshape(-1).tolist()

                if torch.cuda.is_available():
                    data = data.float().cuda()
                    anomaly = anomaly.float().cuda()
                    masks = masks.float().cuda()

                if self.modelType == "mstcn":
                    outputs = self.model(data, masks)
                    outputs = outputs[-1].view(-1)
                elif self.modelType == "mcbtcn":
                    classOutputs, binaryOutputs = self.model(data, masks)
                    outputs = binaryOutputs[-1].view(-1)
                else:
                    outputs = self.model(data)

                mask = (anomaly != self.maskValue).nonzero().squeeze().cpu()
                outputs = outputs[mask]

                clipNames = np.array(clipNames)[mask].tolist()
                anomaly = anomaly[mask]
                loss = self.mseLoss(outputs.squeeze(), anomaly)

                outputs = outputs.reshape(-1).cpu().numpy().tolist()
                anomaly = anomaly.cpu().numpy().flatten().tolist()

                anomalyTargets += anomaly
                anomalyPredictions += outputs

                if step % 10 == 0:
                    progress_bar.set_description("Val [{}]:[{}/{}] Loss: {:.2f}".format(
                        epoch, step, self.valLoader.__len__(), loss.item()))

                for clipName, prediction, target in zip(clipNames, outputs, anomaly):
                    if clipName not in predictions:
                        predictions[clipName] = []
                    if clipName not in targets:
                        targets[clipName] = []
                    predictions[clipName].append(prediction)
                    targets[clipName].append(target)

            videoClips = self.valLoader.dataset.__getVideoClips__()
            for iou in IOUs:
                for s, threshold in enumerate(thresholds):
                    tp, fp, fn = 0, 0, 0
                    normal = {"tp": 0, "fp": 0, "fn": 0}
                    abnormal = {"tp": 0, "fp": 0, "fn": 0}
                    for videoName, clipList in tqdm(videoClips.items()):
                        clipPredictions = []
                        clipTargets = []
                        for clipName in clipList:
                            clipPredictions.append(np.mean(np.array(predictions[clipName])))
                            clipTargets.append(np.mean(np.array(targets[clipName])))
                        # if "Assault010_x264" in videoName:
                        #     auc_score = sklrn.roc_auc_score(clipTargets, clipPredictions)
                        #     utils.visualizeHeatMapPredictions(clipPredictions, clipTargets, self.expFolder, videoName)
                        #     print("AUC Score of selected video: {}".format(auc_score))
                        clipPredictions = (np.array(clipPredictions) > threshold).astype("float32").tolist()
                        if iou == 0.25 and threshold == 0.5:
                            utils.visualizeTemporalPredictions(clipPredictions, clipTargets, self.expFolder, videoName)

                        tp1, fp1, fn1 = f_score(clipPredictions, clipTargets, iou, bg_class=0)
                        abnormal["tp"] += tp1
                        abnormal["fp"] += fp1
                        abnormal["fn"] += fn1

                        tp1, fp1, fn1 = f_score(clipPredictions, clipTargets, iou, bg_class=1)
                        normal["tp"] += tp1
                        normal["fp"] += fp1
                        normal["fn"] += fn1

                        if self.noNormalSegmentation:
                            tp1, fp1, fn1 = f_score(clipPredictions, clipTargets, iou, bg_class=0)
                        else:
                            tp1, fp1, fn1 = f_score(clipPredictions, clipTargets, iou, bg_class=-1)
                            # if "Assault010_x264" in videoName:
                            #     precision = tp1 / float(tp1 + fp1 + 1e-10)
                            #     recall = tp1 / float(tp1 + fn1 + 1e-10)
                            #     f1 = 2.0 * (precision * recall) / (precision + recall + 1e-10)
                            #     print("F1 Score of selected video: {}".format(f1))

                        tp += tp1;
                        fp += fp1;
                        fn += fn1;

                    a_f1, a_precision, a_recall = calc_f1(abnormal["fn"], abnormal["fp"], abnormal["tp"])
                    print('Abnormal F1@%0.2f-%0.2f : %.4f, Precision: %.4f, Recall: %.4f' % (iou, threshold, a_f1,
                                                                                             a_precision * 100,
                                                                                             a_recall * 100))
                    n_f1, n_precision, n_recall = calc_f1(normal["fn"], normal["fp"], normal["tp"])
                    print('Normal F1@%0.2f-%0.2f : %.4f, Precision: %.4f, Recall: %.4f' % (iou, threshold, n_f1,
                                                                                           n_precision * 100,
                                                                                           n_recall * 100))
                    f1, precision, recall = calc_f1(fn, fp, tp)
                    if iou == 0.25 and threshold == 0.5:
                        score = f1
                    print('F1@%0.2f-%0.2f : %.2f, TP: %.2f, FP: %.2f, FN: %.2f' % (iou, threshold, f1, tp, fp, fn))
                    # print('Precision@%0.2f-%0.2f : %.2f, Recall@%0.2f-%0.2f: %.2f' % (iou, threshold, precision * 100,
                    #                                                                   iou, threshold, recall * 100))

                    if self.writer is not None:
                        self.writer.add_scalar("Eval/F1_%0.2f-%0.2f" % (iou, threshold), f1, self.stepCounter)
                        self.writer.add_scalar("Confusion/TP_%0.2f-%0.2f" % (iou, threshold), tp, self.stepCounter)
                        self.writer.add_scalar("Confusion/FP_%0.2f-%0.2f" % (iou, threshold), fp, self.stepCounter)
                        self.writer.add_scalar("Confusion/FN_%0.2f-%0.2f" % (iou, threshold), fn, self.stepCounter)

        fpr, tpr, _ = sklrn.roc_curve(anomalyTargets, anomalyPredictions)
        rocAUC = sklrn.auc(fpr, tpr)
        if self.writer is not None:
            self.writer.add_scalar("Eval/AUC", rocAUC, self.stepCounter)
        print('AUC Score %0.2f' % (rocAUC * 100))

        return score

    def classValidation(self, epoch):
        self.model.eval()
        predictions = {}
        targets = {}

        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            progress_bar = tqdm(self.valLoader)

            for step, (data, masks, anomaly, category, _, clipNames) in enumerate(progress_bar):

                if self.modelType == "mstcn":
                    anomaly = anomaly.view(-1)
                    clipNames = np.array(clipNames).reshape(-1).tolist()

                if torch.cuda.is_available():
                    data = data.cuda().float()
                    anomaly = anomaly.cuda().float()
                    masks = masks.cuda().float()

                outputs = self.model(data, masks)

                loss = 0
                for output in outputs:
                    loss += self.ceLoss(output.transpose(2, 1).reshape(-1, 2), anomaly.long())

                outputs = outputs[-1].transpose(2, 1).reshape(-1, 2).cpu().numpy()
                anomaly = anomaly.cpu().numpy().flatten().tolist()

                if step % 10 == 0:
                    progress_bar.set_description("Val [{}]:[{}/{}] Loss: {:.2f}".format(
                        epoch, step, self.valLoader.__len__(), loss.item()))

                for clipName, prediction, target in zip(clipNames, outputs, anomaly):
                    predictions[clipName] = prediction
                    targets[clipName] = target

            videoClips = self.valLoader.dataset.__getVideoClips__()
            for videoName, clipList in tqdm(videoClips.items()):
                clipPredictions = []
                clipTargets = []
                for clipName in clipList:
                    clipPredictions.append(predictions[clipName])
                    clipTargets.append(targets[clipName])
                clipPredictions = np.argmax(np.array(clipPredictions), axis=1)
                utils.visualizeTemporalPredictions(clipPredictions, clipTargets, self.expFolder, videoName)
                tp1, fp1, fn1 = f_score(clipPredictions, clipTargets, 0.1, bg_class=-1)
                tp += tp1;
                fp += fp1;
                fn += fn1;

        f1, precision, recall = calc_f1(fn, fp, tp)
        print('F1@%0.2f-%0.2f : %.4f, TP: %.4f, FP: %.4f, FN: %.4f' % (0.1, 0.5, f1, tp, fp, fn))
        self.writer.add_scalar("Eval/F1_0.10-%0.2f" % 0.5, f1, self.stepCounter)
        self.writer.add_scalar("Confusion/%0.2f/TP_0.10" % 0.5, tp, self.stepCounter)
        self.writer.add_scalar("Confusion/%0.2f/FP_0.10" % 0.5, fp, self.stepCounter)
        self.writer.add_scalar("Confusion/%0.2f/TP_0.10" % 0.5, fn, self.stepCounter)

        return f1

    def saveCheckpoint(self, path):
        torch.save({"model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "stepCounter": self.stepCounter,
                    "bestScore": self.bestAUC}, path)

    def loadCheckpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.bestAUC = checkpoint["bestScore"]
        self.stepCounter = checkpoint["stepCounter"]


if __name__ == "__main__":
    violenceDetection = ViolenceDetection()
    violenceDetection.train()
