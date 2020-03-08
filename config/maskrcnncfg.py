
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import default_setup

__all__ = ['setup_configs', 'setup_origin_configs']

def setup_configs(args):
    cfg = get_cfg()

    yaml_file = args.config_file

    cfg.merge_from_file(yaml_file)

    cfg.DATASETS.TRAIN = ("davis_val",)
    cfg.DATASETS.TEST = ("davis_val",)  # "davis_val",
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
    # 在这呢,学习率
    cfg.SOLVER.MAX_ITER = 20000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = './output_val'
    cfg.SEED = 3

    cfg.DAVIS_IMG_DIR = '/home/ql-b423/sda/TXH/dataset/davis/JPEGImages/480p'

    cfg.TRAIN_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_train2020.json'
    cfg.VAL_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_val2020.json'
    cfg.TESTDEV_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_test-dev2020.json'
    cfg.TESTCHALLENGE_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_test-challenge2020.json'

    # cfg.freeze()

    default_setup(
        cfg, args
    )
    return cfg


def setup_origin_configs(args):
    cfg = get_cfg()

    yaml_file = args.config_file

    cfg.merge_from_file(yaml_file)

    cfg.SOLVER.CHECKPOINT_PERIOD = args.opts['checkpoint_period']
    cfg.MODEL.MASK_ON = args.opts['MASK_ON']
    cfg.DATASETS.TRAIN = ("davis_val_finetune",)
    # cfg.DATASETS.TRAIN = ("davis_train","davis_val_finetune")
    cfg.DATASETS.TEST = ("davis_val",)  # "davis_val",
    # cfg.DATALOADER.NUM_WORKERS = 4
    if args.opts['pretrain_path'] is None:
        cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS
    else:
        cfg.MODEL.WEIGHTS = args.opts['pretrain_path']  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.opts['batch_size']
    cfg.SOLVER.BASE_LR = args.opts['learning_rate']  # pick a good LR
    cfg.SOLVER.MAX_ITER = args.opts['max_iter']  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.STEPS = args.opts['steps']
    cfg.SOLVER.WARMUP_ITERS = args.opts['warmup_iter']
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    # cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.OUTPUT_DIR = './output_val'
    cfg.SEED = 3
    root_davis = args.opts['root_davis']
    cfg.DAVIS_IMG_DIR = root_davis+'JPEGImages/480p'

    # cfg.TRAIN_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_val-finetune2020.json'
    # cfg.VAL_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_val2020.json'
    # cfg.VAL_FINETUNE_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_val-finetune2020.json'
    # cfg.TESTDEV_DATASET_JSON_PATH =  '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_val2020.json'
    # cfg.TESTCHALLENGE_DATASET_JSON_PATH = '/home/ql-b423/sda/TXH/dataset/davis/COCOAnnos/davis_foreground_ins_test-challenge2020.json'
    cfg.TRAIN_DATASET_JSON_PATH = root_davis + 'COCOAnnos/davis_foreground_ins_train2020.json'
    cfg.VAL_DATASET_JSON_PATH = root_davis + 'COCOAnnos/davis_foreground_ins_val2020.json'
    cfg.VAL_FINETUNE_DATASET_JSON_PATH = root_davis + 'COCOAnnos/davis_foreground_ins_val-finetune2020.json'
    cfg.TESTDEV_DATASET_JSON_PATH = root_davis + 'COCOAnnos/davis_foreground_ins_test-dev2020.json'
    cfg.TESTCHALLENGE_DATASET_JSON_PATH = root_davis + 'COCOAnnos/davis_foreground_ins_test-challenge2020.json'

    # cfg.freeze()

    default_setup(
        cfg, args
    )
    return cfg