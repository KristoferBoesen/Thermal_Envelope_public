"""Configuration loader for the decay heat preprocessor."""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(yaml_path=None) -> Dict[str, Any]:
    """
    Load decay_preprocessor/preprocessor_config.yaml.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Override path.  Defaults to preprocessor_config.yaml in the same
        directory as this file (i.e. decay_preprocessor/preprocessor_config.yaml).

    Returns
    -------
    dict
        Raw configuration values.
    """
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "preprocessor_config.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
