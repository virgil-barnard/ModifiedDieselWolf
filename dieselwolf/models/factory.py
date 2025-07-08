from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Union

import yaml
from torch import nn

from .configurable_cnn import ConfigurableCNN
from .configurable_mobile_rat import ConfigurableMobileRaT


_BACKBONES = {
    "ConfigurableMobileRaT": ConfigurableMobileRaT,
    "ConfigurableCNN": ConfigurableCNN
}


def build_backbone(cfg: Union[str, Path, Mapping[str, Any]]) -> nn.Module:
    """Instantiate a backbone from a YAML config or dictionary.

    Parameters
    ----------
    cfg:
        Either a mapping containing the configuration or a path to a YAML file.

    Returns
    -------
    nn.Module
        Instantiated backbone model.
    """
    if not isinstance(cfg, Mapping):
        with open(Path(cfg), "r") as f:
            cfg_dict: MutableMapping[str, Any] = yaml.safe_load(f)
    else:
        cfg_dict = dict(cfg)

    name = cfg_dict.get("backbone")
    seq_len = cfg_dict.get("seq_len")
    num_classes = cfg_dict.get("num_classes")
    params = cfg_dict.get("params", {})

    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'")
    if seq_len is None or num_classes is None:
        raise ValueError("'seq_len' and 'num_classes' must be specified")

    cls = _BACKBONES[name]
    return cls(seq_len=seq_len, num_classes=num_classes, **params)
