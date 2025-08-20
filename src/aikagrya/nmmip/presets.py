PRESETS = {
    "128": dict(dim=128, blocks=16, steps=300_000, trials=10, seed=1234),
    "256": dict(dim=256, blocks=32, steps=400_000, trials=10, seed=2234),
    "512": dict(dim=512, blocks=64, steps=500_000, trials=10, seed=3234),
    "1080": dict(dim=1080, blocks=60, steps=700_000, trials=5, seed=4234),
}

DEFAULT = dict(
    eps=5e-6,
    alpha_warm=0.60,
    alpha_mid=0.95,
    alpha_end=0.9995,
    tau_start=0.12,
    tau_end=0.02,
    lifts_on=True,
    clamp_u=0.08,
    output="runs/nmmip",
    verbose=False,
)


