"""Configuration loading shared by every stage."""
import os
from pathlib import Path
from typing import Optional, Union

import yaml

DEFAULT_CONFIG = "configs/config.yaml"


def load_config(path: Optional[Union[str, Path]] = None) -> dict:
    """Load YAML config. Priority: explicit path > $BIOFORGE_CONFIG > configs/config.yaml."""
    path = path or os.environ.get("BIOFORGE_CONFIG", DEFAULT_CONFIG)
    with open(path, "r") as f:
        return yaml.safe_load(f)
