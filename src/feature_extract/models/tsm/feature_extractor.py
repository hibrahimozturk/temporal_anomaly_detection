import cv2
from feature_extract.models.tsm.ops.models import TSN
from feature_extract.models.tsm.ops.transforms import *
from PIL import Image


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


class TSMFeatureExtractor:
    def __init__(self, weightPath, segments, crops, fullSize=True):
        self.weightPath = weightPath
        self.segments = segments
        self.crops = crops
        self.fullSize = fullSize

        self.is_shift, shift_div, shift_place = parse_shift_option_from_log_name(self.weightPath)
        if 'RGB' in self.weightPath:
            self.modality = 'RGB'
        else:
            self.modality = 'Flow'
        this_arch = self.weightPath.split('TSM_')[1].split('_')[2]

        self.num_class = 400
        print('=> shift: {}, shift_div: {}, shift_place: {}'.format(self.is_shift, shift_div, shift_place))
        self.net = TSN(self.num_class, self.segments if self.is_shift else 1, self.modality,
                       base_model=this_arch,
                       consensus_type="avg",
                       img_feature_dim=256,
                       pretrain="imagenet",
                       is_shift=self.is_shift, shift_div=shift_div, shift_place=shift_place,
                       non_local='_nl' in self.weightPath,
                       )

        if 'tpool' in self.weightPath:
            from ops.temporal_shift import make_temporal_pool
            make_temporal_pool(self.net.base_model, self.segments)  # since DataParallel

        checkpoint = torch.load(self.weightPath)
        checkpoint = checkpoint['state_dict']

        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                        'base_model.classifier.bias': 'new_fc.bias',
                        }

        for k, v in replace_dict.items():
            if k in base_dict:
                base_dict[v] = base_dict.pop(k)

        self.net.load_state_dict(base_dict)

        input_size = self.net.scale_size if self.fullSize else self.net.input_size
        if self.crops == 1:
            cropping = torchvision.transforms.Compose([
                GroupScale(self.net.scale_size),
                GroupCenterCrop(input_size),
            ])
        elif self.crops == 3:  # do not flip, so only 5 crops
            cropping = torchvision.transforms.Compose([
                GroupFullResSample(input_size, self.net.scale_size, flip=False)
            ])
        elif self.crops == 5:  # do not flip, so only 5 crops
            cropping = torchvision.transforms.Compose([
                GroupOverSample(input_size, self.net.scale_size, flip=False)
            ])
        elif self.crops == 10:
            cropping = torchvision.transforms.Compose([
                GroupOverSample(input_size, self.net.scale_size)
            ])
        else:
            raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(self.crops))

        self.transform = torchvision.transforms.Compose([
            cropping,
            Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
            GroupNormalize(self.net.input_mean, self.net.input_std)])

        if self.modality == 'RGB':
            self.length = 3
        elif self.modality == 'Flow':
            self.length = 10
        elif self.modality == 'RGBDiff':
            self.length = 18

        self.net = self.net.cuda()
        self.net.eval()

    def feature_extract(self, inputBatch):
        assert len(inputBatch[0]) == self.segments, "{} frames should be in clip not {}".format(self.segments,
                                                                                                len(inputBatch))
        with torch.no_grad():
            data = []
            for inputFrames in inputBatch:
                data.append(torch.tensor(self.transform(inputFrames)))
            # data = self.transform(inputBatch)
            data = torch.stack(data)
            data = data.cuda()
            data_in = data.view(-1, self.length, data.size(2), data.size(3))

            if self.is_shift:
                data_in = data_in.view(len(inputBatch) * self.crops, self.segments,
                                       self.length, data_in.size(2), data_in.size(3))

            rst, base_out, features = self.net(data_in)
            features = features.reshape(len(inputBatch), self.crops, self.segments, -1).mean(1)

        return features


if __name__ == "__main__":
    extractor = TSMFeatureExtractor(
        "pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth",
        segments=8, crops=3, batchSize=1)

    t_stride = 64 // 8

    capture = cv2.VideoCapture("test2.mp4")
    inputs = []
    i = 0

    while True:
        ret, img = capture.read()
        if not ret:
            break
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img).convert("RGB")
        if i % t_stride == 0:
            inputs.append(im_pil)

        i += 1
        if len(inputs) == 8:
            extractor.feature_extract(inputs)
            inputs = []
