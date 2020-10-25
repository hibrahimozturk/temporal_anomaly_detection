from feature_extract.batch_process.BatchProcessor import BatchProcessor

from keras.models import model_from_json
from keras import backend as K
import numpy as np
import queue


class C3DProcessor(BatchProcessor):
    def __init__(self, model_json, model_weight, batch: queue.Queue,
                 outputs: queue.Queue, dry_run: bool):
        BatchProcessor.__init__(self, batch, outputs, dry_run)
        self.model = self.__prepare_model(model_json, model_weight)
        return

    @staticmethod
    def __prepare_model(model_json, model_weight):
        # override backend if provided as an input arg
        backend = 'tf'
        print("[Info] Using backend={}".format(backend))

        print("[Info] Reading model architecture...")
        model = model_from_json(open(model_json, 'r').read())
        # model = c3d_model.get_model(backend=backend)

        print("[Info] Loading model weights...")
        model.load_weights(model_weight)

        inp = model.input  # input placeholder
        outputs = [layer.output for layer in model.layers if layer.name == "fc6"]  # all layer outputs
        functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]  # evaluation functions

        # Testing
        test = np.random.random((1, 16, 112, 112, 3))
        layer_outs = [func([test, 1.]) for func in functors]
        print(layer_outs)

        # int_model = c3d_model.get_int_model(model=model, layer='fc6', backend=backend)
        int_model = functors[0]
        return int_model

    def process_batch(self):
        if not self.dry_run:
            out_tensor = self.model([self.local_batch, 1])[0]
            out_tensor = out_tensor / np.linalg.norm(out_tensor)
        else:
            out_tensor = np.zeros((self.local_batch.shape[0], 10))
        return out_tensor

    def put_queue(self, output):
        assert output.shape[0] == len(self.clip_names) == len(self.video_names) == len(self.targets), \
            "# of elements are not same"
        for i in range(output.shape[0]):
            self.outputs.put({"out_tensor": output[i],
                              "clip_name": self.clip_names[i],
                              "video_name": self.video_names[i],
                              "anomaly": self.targets[i]})
