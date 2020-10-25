model = dict(
    num_stages=5,
    num_layers=6,
    hidden_win_size=64,
    feature_size=1024,
    first_stage_repeat=1,
    losses=[
        dict(type="mse", factor=1),
        dict(type="thp", factor=0.5,
             params=dict(
                 margin=0.55,
                 measure="output",
                 max_violation=True))
    ]
)
