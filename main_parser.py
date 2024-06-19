import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training Settings and Hyperparameters')

    # Basic settings
    parser.add_argument('--task', type=str, help='YOLO task, i.e. detect, segment, classify, pose')
    parser.add_argument('--mode',default="predict" ,type=str, help='YOLO mode, i.e. train, val, predict, export, track, benchmark')

    # Train settings
    parser.add_argument('--model',type=str, help='path to model file, i.e. yolov8n.pt, yolov8n.yaml')
    parser.add_argument('--data', type=str, help='path to data file, i.e. coco8.yaml')
    parser.add_argument('--epochs', type=int, help='number of epochs to train for')
    parser.add_argument('--time', type=float, help='number of hours to train for, overrides epochs if supplied')
    parser.add_argument('--patience', type=int, help='epochs to wait for no observable improvement for early stopping of training')
    parser.add_argument('--batch', type=int, help='number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=lambda s: [int(item) for item in s.split(',')], help='input images size as int for train and val modes, or list[w,h] for predict and export modes')
    parser.add_argument('--save', type=bool, help='save train checkpoints and predict results')
    parser.add_argument('--save_period', type=int, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', type=lambda s: s if s.lower() in ['true', 'false', 'ram', 'disk'] else False, help='Use cache for data loading')
    parser.add_argument('--device', type=str, help='device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu')
    parser.add_argument('--workers', type=int, help='number of worker threads for data loading (per RANK if DDP)')
    parser.add_argument('--project', type=str, help='project name')
    parser.add_argument('--name', type=str, help='experiment name, results saved to \'project/name\' directory')
    parser.add_argument('--exist_ok', type=bool, help='whether to overwrite existing experiment')
    parser.add_argument('--pretrained', type=lambda s: s if s.lower() in ['true', 'false'] else str, help='whether to use a pretrained model (bool) or a model to load weights from (str)')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help='optimizer to use')
    parser.add_argument('--verbose', type=bool, help='whether to print verbose output')
    parser.add_argument('--seed', type=int, help='random seed for reproducibility')
    parser.add_argument('--deterministic', type=bool, help='whether to enable deterministic mode')
    parser.add_argument('--single_cls', type=bool, help='train multi-class data as single-class')
    parser.add_argument('--rect', type=bool, help='rectangular training if mode=\'train\' or rectangular validation if mode=\'val\'')
    parser.add_argument('--cos_lr', type=bool, help='use cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, help='disable mosaic augmentation for final epochs (0 to disable)')
    parser.add_argument('--resume', type=bool, help='resume training from last checkpoint')
    parser.add_argument('--amp', type=bool, help='Automatic Mixed Precision (AMP) training')
    parser.add_argument('--fraction', type=float, help='dataset fraction to train on (default is 1.0, all images in train set)')
    parser.add_argument('--profile', type=bool, help='profile ONNX and TensorRT speeds during training for loggers')
    parser.add_argument('--freeze', type=lambda s: [int(item) for item in s.split(',')], help='freeze first n layers, or freeze list of layer indices during training')
    parser.add_argument('--multi_scale', type=bool, help='Whether to use multiscale during training')
    parser.add_argument('--overlap_mask', type=bool, help='masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, help='mask downsample ratio (segment train only)')
    parser.add_argument('--dropout', type=float, help='use dropout regularization (classify train only)')

    # Val/Test settings
    parser.add_argument('--val', type=bool, help='validate/test during training')
    parser.add_argument('--split', type=str, choices=['val', 'test', 'train'], help='dataset split to use for validation')
    parser.add_argument('--save_json', type=bool, help='save results to JSON file')
    parser.add_argument('--save_hybrid', type=bool, help='save hybrid version of labels (labels + additional predictions)')
    parser.add_argument('--conf', type=float, help='object confidence threshold for detection (default 0.25 predict, 0.001 val)')
    parser.add_argument('--iou', type=float, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, help='maximum number of detections per image')
    parser.add_argument('--half', type=bool, help='use half precision (FP16)')
    parser.add_argument('--dnn', type=bool, help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', type=bool, help='save plots and images during train/val')

    # Predict settings
    parser.add_argument('--source', type=str, help='source directory for images or videos')
    parser.add_argument('--vid_stride', type=int, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', type=bool, help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--visualize', type=bool, help='visualize model features')
    parser.add_argument('--augment', type=bool, help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', type=bool, help='class-agnostic NMS')
    parser.add_argument('--classes', type=lambda s: [int(item) for item in s.split(',')], help='filter results by class, i.e. classes=0, or classes=[0,2,3]')
    parser.add_argument('--retina_masks', type=bool, help='use high-resolution segmentation masks')
    parser.add_argument('--embed', type=lambda s: [int(item) for item in s.split(',')], help='return feature vectors/embeddings from given layers')
    parser.add_argument('--done_warmup', type=bool, help='done warmup')
    parser.add_argument('--webcam',default=False,type=bool,help="provide the link for IP cam else uses system cam")

    # Visualize settings
    parser.add_argument('--show', type=bool, help='show predicted images and videos if environment allows')
    parser.add_argument('--save_frames', type=bool, help='save predicted individual video frames')
    parser.add_argument('--save_txt', type=bool, help='save results as .txt file')
    parser.add_argument('--save_conf', type=bool, help='save results with confidence scores')
    parser.add_argument('--save_crop', type=bool, help='save cropped images with results')
    parser.add_argument('--show_labels', type=bool, help='show prediction labels, i.e. \'person\'')
    parser.add_argument('--show_conf', type=bool, help='show prediction confidence, i.e. \'0.99\'')
    parser.add_argument('--show_boxes', type=bool, help='show prediction boxes')
    parser.add_argument('--line_width', type=int, help='line width of the bounding boxes. Scaled to image size if None')

    # Hyperparameters
    parser.add_argument('--lr0', type=float, help='initial learning rate (i.e. SGD=1E-2, Adam=1E-3)')
    parser.add_argument('--lrf', type=float, help='final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, help='optimizer weight decay 5e-4')
    parser.add_argument('--warmup_epochs', type=float, help='warmup epochs (fractions ok)')
    parser.add_argument('--warmup_momentum', type=float, help='warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, help='warmup initial bias lr')
    parser.add_argument('--box', type=float, help='box loss gain')
    parser.add_argument('--cls', type=float, help='cls loss gain (scale with pixels)')
    parser.add_argument('--dfl', type=float, help='dfl loss gain')
    parser.add_argument('--pose', type=float, help='pose loss gain')
    parser.add_argument('--kobj', type=float, help='keypoint obj loss gain')
    parser.add_argument('--label_smoothing', type=float, help='label smoothing (fraction)')
    parser.add_argument('--nbs', type=int, help='nominal batch size')
    parser.add_argument('--hsv_h', type=float, help='image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv_s', type=float, help='image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv_v', type=float, help='image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, help='image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, help='image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, help='image scale (+/- gain)')
    parser.add_argument('--shear', type=float, help='image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, help='image perspective (+/- fraction), range 0-0.001')
    parser.add_argument('--flipud', type=float, help='image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, help='image flip left-right (probability)')
    parser.add_argument('--bgr', type=float, help='image channel BGR (probability)')
    parser.add_argument('--mosaic', type=float, help='image mosaic (probability)')
    parser.add_argument('--mixup', type=float, help='image mixup (probability)')
    parser.add_argument('--copy_paste', type=float, help='segment copy-paste (probability)')
    parser.add_argument('--auto_augment', type=str, choices=['randaugment', 'autoaugment', 'augmix'], help='auto augmentation policy for classification')
    parser.add_argument('--erasing', type=float, help='probability of random erasing during classification training (0-0.9)')
    parser.add_argument('--crop_fraction', type=float, help='image crop fraction for classification (0.1-1)')

    # Custom config.yaml
    parser.add_argument('--cfg', type=str, help='for overriding defaults.yaml')

    # Tracker settings
    parser.add_argument('--tracker', type=str, choices=['botsort.yaml', 'bytetrack.yaml'], help='tracker type')

    args = parser.parse_args()

    # Convert args to dictionary and filter out None values
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    return args_dict

if __name__ == "__main__":
    args_dict = parse_args()
    print(args_dict)
