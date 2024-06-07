import os
from pathlib import Path
import sys

from cfg import RANK, ROOT, RUNS_DIR, config
from data.build import build_yolo_dataset
from data.datset import check_det_dataset
from utils.yaml_util import yaml_print, yaml_save

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str, pathlib.Path): Path to increment.
        exist_ok (bool, optional): If True, the path will not be incremented and returned as-is. Defaults to False.
        sep (str, optional): Separator to use between the path and the incrementation number. Defaults to ''.
        mkdir (bool, optional): Create a directory if it does not exist. Defaults to False.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path
def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        project = args.project or (ROOT/RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir)

class BaseTrainer:
    def __init__(self,cfg,overrides=None):
        self.args=config.get_cfg(cfg,overrides)
                # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
        self.args.save_dir = str(self.save_dir)
        yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths

        self.batch_size = self.args.batch   
        self.epochs = self.args.epochs
        self.start_epoch = 0


        #Dataset self.trainset is imgpath
        self.trainset, self.testset = self.get_dataset()
        self.ema = None
        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        
    

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            data = check_det_dataset(self.args.data)
        except Exception as e:
            raise RuntimeError((f"Dataset '{(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")
    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = 32
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)

    
