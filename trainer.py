import os
import torch
import logging

import detectron2.utils.comm as comm

from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel

from config.maskrcnncfg import setup_origin_configs


from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer

from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset, print_csv_format

from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from datasets.regist_datasets import regist_datasets
from visual_v1 import visual

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    aaa = MetadataCatalog.get(dataset_name)
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get('iteration', -1) + 1
    )

    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, 'metric.json')),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    data_loader = build_detection_train_loader(cfg)
    logger.info(" Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                k: v.item()
                for k,v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST_EVAL_PERIOC == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def trainer(args,visual_,visual_threshold):
    cfg = setup_origin_configs(args)
    print('cfg',)
    regist_datasets(cfg)
    model = build_model(cfg)

    if visual_:
        # 不设置就是0.05
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=visual_threshold
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.best_model_name)
        visual(cfg)
        return None

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        # load pretrained model weights:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.best_model_name)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model)
    return do_test(cfg, model)

