_base_ = ["../configs/model_cfg.py", "../configs/dataset_cfg.py"]

model_type = "mstcn"
log_step = 5

train = dict(
    num_epochs=10,
    exp_dir="../exps/MSTCN/10-w64-s5-l6",
    tensorboard=True,
    optimizer=dict(
        lr=1e-4,
        scheduler=dict(step_size=10, gamma=1e-1),
    )
)

evaluation = dict(
    thresholds=[0.50, 0.75, 0.90],
    iou_list=[0.25, 0.50, 0.75]
)
