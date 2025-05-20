import yaml
import logging
from pathlib import Path
from typing import Any, Dict

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s %(levelname)s ▶ %(message)s',
        level=logging.INFO
    )
