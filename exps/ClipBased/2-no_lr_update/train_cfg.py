_base_ = ["configs/model_cfg.py", "configs/dataset_cfg.py"]

model_type = "mlp"
log_step = 5

dataset = dict(
    batch_size=64
)

train = dict(
    num_epochs=10,
    exp_dir="../exps/ClipBased/2-no_lr_update",
    tensorboard=True,
    optimizer=dict(
        lr=1e-4,
        scheduler=dict(step_size=10, gamma=1e-1)
    )
)
