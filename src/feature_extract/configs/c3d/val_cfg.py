_base_ = "../with_extras_cfg.py"

extractor_type = "c3d"

model = dict(
    json="models/c3d/sports1M_weights_tf.json",
    weight="models/c3d/sports1M_weights_tf.h5"
)

input_processor = dict(
    temporal_stride=16,
    input_length=16,
    batch_size=5,
    input_mean="models/c3d/train01_16_128_171_mean.npy",
    input_size=(112, 112)
)

output_writer = dict(
    clip_folder="../../data/c3d_features/abnormal/val_test",
    json_path="../../data/c3d_features/abnormal/val_test/ValLabels.json"
)

extractor = dict(
    temporal_annotions="../../data/Annotations/Splits/abnormal/val.json",
    video_folder="../../data/Videos",
    num_producers=5,
    dry_run=False,
    top_k=5
)



