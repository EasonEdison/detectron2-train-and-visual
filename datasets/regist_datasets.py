from detectron2.data.datasets import register_coco_instances


def regist_datasets(cfg):
    register_coco_instances("davis_train", {}, cfg.TRAIN_DATASET_JSON_PATH, cfg.DAVIS_IMG_DIR)
    register_coco_instances("davis_val", {}, cfg.VAL_DATASET_JSON_PATH, cfg.DAVIS_IMG_DIR)
    register_coco_instances("davis_val_finetune", {}, cfg.VAL_FINETUNE_DATASET_JSON_PATH, cfg.DAVIS_IMG_DIR)
    register_coco_instances("davis_test-dev", {}, cfg.TESTDEV_DATASET_JSON_PATH, cfg.DAVIS_IMG_DIR)
    register_coco_instances("davis_test-challenge", {}, cfg.TESTCHALLENGE_DATASET_JSON_PATH, cfg.DAVIS_IMG_DIR)