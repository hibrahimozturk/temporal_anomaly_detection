_base_ = "../with_extras_cfg.py"

extractor_type = "i3d"

model = dict(
    path="models/i3d/model_rgb.pth"
)

input_processor = dict(
    temporal_stride=16,
    input_length=79,
    batch_size=5,
    input_size=(224, 224)
)

output_writer = dict(
    clip_folder="../../data/extracted_features/i3d/minival/abnormal/clips",
    json_path="../../data/extracted_features/i3d/minival/abnormal/minival.json"
)

extractor = dict(
    temporal_annotions="../../data/Annotations/Splits/abnormal/val.json",
    video_folder="../../data/Videos",
    num_producers=5,
    dry_run=False,
    top_k=10
)
