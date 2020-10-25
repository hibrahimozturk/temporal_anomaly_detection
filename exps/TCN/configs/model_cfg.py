model = dict(
    kernel_size=21,
    feature_size=1024,
    losses=[
        dict(type="mse", factor=1),
        dict(type="thp", factor=0.5,
             params=dict(margin=0.55,
                         measure="output",
                         max_violation=True))
    ]
)
