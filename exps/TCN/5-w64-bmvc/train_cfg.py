_base_ = ["../configs/model_cfg.py", "../configs/dataset_cfg.py"]

model_type = "edtcn"
log_step = 5

train = dict(
    num_epochs=10,
    exp_dir="../exps/TCN/5-w64-bmvc",
    tensorboard=True,
    optimizer=dict(
        lr=1e-4,
        scheduler=dict(step_size=10, gamma=1e-1)
    )
)
