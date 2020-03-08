from detectron2.config import get_cfg
from detectron2.engine import default_setup

from instances_a.pointrend.point_rend.config import add_pointrend_config

__all__ = ['setup_origin_configs']


def setup_origin_configs(args):
    cfg = get_cfg()
    # cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True
    add_pointrend_config(cfg)

    yaml_file = args.config_file

    cfg.merge_from_file(yaml_file)

    cfg.DATASETS.TRAIN = ("davis_train","davis_val_finetune")
    cfg.DATASETS.TEST = ("davis_val",)  # "davis_val",
    cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.MAX_ITER = 100
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = './output_val'
    cfg.SEED = 3

    cfg.DAVIS_IMG_DIR = '/root/dataset/davis/JPEGImages/480p'

    cfg.TRAIN_DATASET_JSON_PATH = '/root/dataset/davis/COCOAnnos/davis_foreground_ins_train2020.json'
    cfg.VAL_DATASET_JSON_PATH = '/root/dataset/davis/COCOAnnos/davis_foreground_ins_val2020.json'
    cfg.VAL_FINETUNE_DATASET_JSON_PATH = '/root/dataset/davis/COCOAnnos/davis_foreground_ins_val-finetune2020.json'
    cfg.TESTDEV_DATASET_JSON_PATH = '/root/dataset/davis/COCOAnnos/davis_foreground_ins_test-dev2020.json'
    cfg.TESTCHALLENGE_DATASET_JSON_PATH = '/root/dataset/davis/COCOAnnos/davis_foreground_ins_test-challenge2020.json'

    # cfg.freeze()

    default_setup(
        cfg, args
    )
    return cfg