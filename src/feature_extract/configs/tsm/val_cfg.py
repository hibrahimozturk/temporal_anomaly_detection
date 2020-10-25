_base_ = "../with_extras_cfg.py"

extractor_type = "tsm"
input_length = 8
model = dict(
    path="models/tsm/pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth",
    input_length=input_length,
    num_crops=1
)

input_processor = dict(
    input_length=input_length,
    batch_size=24,
    input_size=(256, 256),
)

output_writer = dict(
    clip_folder="../../../data/tsm_features/1crop/abnormal/val",
    json_path="../../data/tsm_features/1crop/abnormal/ValLabels.json"
)

extractor = dict(
    temporal_annotions="../../data/Annotations/Splits/abnormal/val.json",
    video_folder="../../data/Videos",
    num_producers=5,
    dry_run=True
)
