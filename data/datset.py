import contextlib
from copy import deepcopy
import glob
import hashlib
from itertools import repeat
import math
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from data import CACHE_VERSION, IMG_FORMATS
from data.augment import Compose, Format, LetterBox, v8_transforms
from data.instance import Instances
from data.ops import resample_segments, segments2boxes
from utils import LOGGER, NUM_THREADS, TQDM
from utils.yaml_util import yaml_load

from cfg import DATASETS_DIR, DEFAULT_CFG, LOCAL_RANK, ROOT


def check_file(file, suffix="", download=True, hard=True):
    files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))  # find file
    if not files and hard:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(files) > 1 and hard:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
    return files[0] if len(files) else []  # return file

def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """

    file = check_file(dataset)

    # Read YAML
    data = yaml_load(file, append_filename=True)  # dictionary

    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    (f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.warning("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError((f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError((f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    # Resolve paths
    path = Path(data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve()

    # Set paths
    data["path"] = path  # download scripts
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    return data  # dictionary





class YOLODataset(Dataset):
    def __init__(
        self,
        img_path,
        data=None,
        imgsz=640,
        augment=True,
        hyp=DEFAULT_CFG,
        task='detect',
        prefix="",
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
    ):
        """Initialize BaseDataset with given configuration and options."""
        super().__init__()
        self.single_cls=False,
        self.data = data
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        # self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        self.cache=None
        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)
    def update_labels(self, include_class: Optional[list]):
        """Update labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def get_img_files(self,img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"No images found in {img_path}."
        except Exception as e:
            raise FileNotFoundError(f"Error loading data from {img_path}\n") from e
        return im_files

    def img2label_paths(self,img_paths):
        """Define label paths as a function of image paths."""
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

    def get_hash(self,paths):
        """Returns a single hash value of a list of paths (files or dirs)."""
        size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
        h = hashlib.sha256(str(size).encode())  # hash sizes
        h.update("".join(paths).encode())  # hash paths
        return h.hexdigest()  # return hash
    def load_dataset_cache_file(self,path):
        """Load an   *.cache dictionary from path."""
        import gc

        gc.disable()  # reduce pickle load time https://github.com/ / /pull/1585
        cache = np.load(str(path), allow_pickle=True).item()  # load dict
        gc.enable()
        return cache
    def exif_size(img: Image.Image):
        """Returns exif-corrected PIL size."""
        s = img.size  # (width, height)
        if img.format == "JPEG":  # only support JPEG images
            with contextlib.suppress(Exception):
                exif = img.getexif()
                if exif:
                    rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                    if rotation in {6, 8}:  # rotation 270 or 90
                        s = s[1], s[0]
        return s

    def verify_image_label(self,args):
        """Verify one image-label pair."""
        im_file, lb_file, prefix, num_cls= args
        # Number (missing, found, empty, corrupt), message, segments, keypoints
        nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
        try:
            # Verify images
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = self.exif_size(im)  # image size
            shape = (shape[1], shape[0])  # hw
            assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
            assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}."
            if im.format.lower() in {"jpg", "jpeg"}:
                with open(im_file, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                        msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

            # Verify labels
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    lb = np.array(lb, dtype=np.float32)
                nl = len(lb)
                if nl:
                    assert lb.shape[1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                    assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                    assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                    # All labels
                    max_cls = lb[:, 0].max()  # max label count
                    assert max_cls <= num_cls, (
                        f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                        f"Possible class labels are 0-{num_cls - 1}"
                    )
                    _, i = np.unique(lb, axis=0, return_index=True)
                    if len(i) < nl:  # duplicate row check
                        lb = lb[i]  # remove duplicates
                        if segments:
                            segments = [segments[x] for x in i]
                        msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
                else:
                    ne = 1  # label empty
                    lb = np.zeros((0,5), dtype=np.float32)
            else:
                nm = 1  # label missing
                lb = np.zeros((0,5), dtype=np.float32)
            lb = lb[:, :5]
            return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
        except Exception as e:
            nc = 1
            msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
            return [None, None, None, None, None, nm, nf, ne, nc, msg]

    def save_dataset_cache_file(prefix, path, x, version):
        """Save an   dataset *.cache dictionary x to path."""
        x["version"] = version  # add cache version
        if os.access(str(path.parent), os.W_OK):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{prefix}New cache created: {path}")
        else:
            LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=self.verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(len(self.data["names"])),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": lb[:, 0:1],  # n, 1
                            "bboxes": lb[:, 1:],  # n, 4
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}.")
        x["hash"] = self.get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        self.save_dataset_cache_file(self.prefix, path, x, CACHE_VERSION)
        return x
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = self.img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = self.load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == CACHE_VERSION  # matches current version
            assert cache["hash"] == self.get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly.")
        return labels

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(index))
    def get_image_and_label(self, index):
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ / /pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        # if self.rect:
        #     label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)
    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        label["instances"] = Instances(bboxes,bbox_format=bbox_format, normalized=normalized)
        return label
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}WARNING ⚠️ Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = cv2.imread(f)  # BGR
            else:  # read image
                im = cv2.imread(f)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]
    

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"masks", "bboxes", "cls"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
    def __len__(self):
        return len(self.labels)
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment else 0.0
            hyp.mixup = hyp.mixup if self.augment else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms
    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)