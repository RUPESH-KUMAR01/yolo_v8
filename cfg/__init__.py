import os
from pathlib import Path
# Now you can import from the 'utils' module
from utils import IterableSimpleNamespace
from utils.yaml_util import yaml_load




LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1)) 
RANK = int(os.getenv("RANK", -1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
#specific to device
RUNS_DIR=r"runs"
DATASETS_DIR=r"C:\Users\thata\intern\code\pre-built-models\datasets"
ASSETS=Path(r"C:\Users\thata\intern\code\pre-built-models\modified\assets")
# MODEL_PAR=r"C:\Users\thata\intern\code\pre-built-models\modified\cfg\model.yaml"
weights_dir=r"weights"
