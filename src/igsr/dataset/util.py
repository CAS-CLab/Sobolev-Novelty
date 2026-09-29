"""Offline dispatch restricted to SRBench black-box regression."""


def get_dataset(cfg):
    if cfg.name == "srbench_blackbox_local_controlled":
        from igsr.dataset.real.srbench_blackbox import load_srbench_blackbox_dataset

        return load_srbench_blackbox_dataset(cfg)
    raise ValueError(f"Unsupported review dataset: {cfg.name}")
