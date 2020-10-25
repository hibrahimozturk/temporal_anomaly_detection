_base_ = ["../configs/model_cfg.py", "../configs/dataset_cfg.py"]

model_type = "mstcn"
load_from = "../exps/MSTCN/10-w64-s5-l6/checkpoints/epoch-1.pth"
log_step = 5

evaluation = dict(
    thresholds=[0.50, 0.75, 0.90],
    iou_list=[0.25, 0.50, 0.75]
)
