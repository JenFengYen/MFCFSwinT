#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.
This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""
#This file has been modified by the author
import logging
import os,sys
import numpy as np
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage

from detectron2.data.datasets import register_coco_instances
from convert_torchvision_to_d2_v2_fun import swinT_weight_to_dt2
import pickle
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params

import itertools
import backbone.swin_transformer_pkd
logger = logging.getLogger("detectron2")
from detectron2_edit.data.dataset_mapper import DatasetMapper
from detectron2.utils.logger import _log_api_usage, log_first_n
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.config import configurable
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset
from detectron2.data.build import build_batch_data_loader
import torch.utils.data as torchdata
from detectron2.data import DatasetCatalog, MetadataCatalog
import backbone.meta_arch.rcnn
def _train_loader_Edge_data_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_train_loader_Edge_data_from_config)
def build_detection_train_loader_Edge_data(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

def build_optimizer(cfg, model):
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )

    def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
        # detectron2 doesn't have full model gradient clipping now
        clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        enable = (
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED
            and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
            and clip_norm_val > 0.0
        )

        class FullModelGradientClippingOptimizer(optim):
            def step(self, closure=None):
                all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                super().step(closure=closure)

        return FullModelGradientClippingOptimizer if enable else optim

    optimizer_type = cfg.SOLVER.OPTIMIZER
    if optimizer_type == "SGD":
        optimizer = maybe_add_gradient_clipping(torch.optim.SGD)(
            params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM,
            nesterov=cfg.SOLVER.NESTEROV,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif optimizer_type == "AdamW":
        optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
            params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"no optimizer type {optimizer_type}")
    return optimizer


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
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
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
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    if 'EDGE' in cfg.MODEL.keys():
        if cfg.MODEL.EDGE:
            data_loader = build_detection_train_loader_Edge_data(cfg)
    else:
        data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    best_ap = 0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            if 'MASK_LOSS_FILTER' in cfg.MODEL:
                loss_dict['loss_mask'] *=cfg.MODEL.MASK_LOSS_FILTER
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict
            
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            if iteration == cfg.TEST.EVAL_PERIOD_START_ITER - 1 or(
                iteration > cfg.TEST.EVAL_PERIOD_START_ITER - 1
                and cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1-cfg.TEST.EVAL_PERIOD_START_ITER) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                result = do_test(cfg, model)
                for key in result.keys():
                    if 'segm' == key:
                        ap = result[key]['AP']
                        if ap > best_ap : 
                            best_ap = ap
                            additional_state = {"iteration": iteration}
                            periodic_checkpointer.save('model_best',**additional_state)
                            print("save best segm ap",ap)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
                

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

from collections import Counter
import tqdm
from detectron2.utils.analysis import  FlopCountAnalysis
from fvcore.nn import flop_count_table
def do_flop(cfg,model):
    if 'EDGE' in cfg.MODEL.keys():
        if cfg.MODEL.EDGE:
            data_loader = build_detection_train_loader_Edge_data(cfg)
    else:
        data_loader = build_detection_train_loader(cfg)

    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    counts = Counter()
    total_flops = []
    for idx, data in zip(tqdm.trange(cfg.MODEL.DATASET_COUNT), data_loader):  # noqa
        flops = FlopCountAnalysis(model, data)
        if idx > 0:
            flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        counts += flops.by_operator()
        total_flops.append(flops.total())

    logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    logger.info(
        "Average GFlops for each type of operators:\n"
        + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    )
    logger.info(
        "Total GFlops: {:.1f}+-{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    )
def do_flop_thop(cfg,model):
    if 'EDGE' in cfg.MODEL.keys():
        if cfg.MODEL.EDGE:
            data_loader = build_detection_train_loader_Edge_data(cfg)
    else:
        data_loader = build_detection_train_loader(cfg)

    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    import torch
    from torchvision import models
    from thop.profile import profile
    import torchvision.models as models
    from thop import clever_format
    total_flops = []
    for idx, data in zip(tqdm.trange(1), data_loader):  # noqa
        macs, params = profile(model, inputs=(data, ))
        flops = macs*2

        total_flops.append(flops)
        flops, params = clever_format([flops, params], "%.3f")
        print("FLOPs="+flops)
    print("Total FLOPs = "+ str(np.mean(total_flops)))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg
def setup_cfg(args):
    cfg = get_cfg() 
    epoch = 200
    shut_down_restore_iter = -1
    
    model_list=[
    'swint_T_MFFv3_PKD_Edegev3_CascadeROIHeads_pretrain_imagenet1K_freeze_3_lr_0.0001_step_50_60_batch_5_do_flops',
    'swint_T_pretrain_imagenet1K_freeze_3_lr_0.0001_step_50_60_batch_5_do_flops',
    'swint_T_PKD_pretrain_imagenet1K_freeze_3_lr_0.0001_step_50_60_batch_5_do_flops',
    'swint_T_MFFv3_PKD_pretrain_imagenet1K_freeze_3_lr_0.0001_step_50_60_batch_5_do_flops',
    'swint_T_MFFv3_PKD_Edegev3_pretrain_imagenet1K_freeze_3_lr_0.0001_step_50_60_batch_5_do_flops',
    ]
    model_name = model_list[0]
    model_setting_name = model_name
    from configs.SwinT.config import add_swint_config_tiny
    if model_name==model_list[0]:
        add_swint_config_tiny(cfg)
        cfg.merge_from_file("./detectron2_code/configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")
        pretrain_dir = "./detectron2_code/pretrain/"
        pretrain_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224_dt2.pth")
        if not os.path.isfile(pretrain_weight_dir):
            orginal_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224.pth")
            if os.path.isfile(orginal_weight_dir):
                swinT_weight_to_dt2(orginal_weight_dir)
            else:
                print('Did not use pretrain weight')
        cfg.MODEL.WEIGHTS = pretrain_weight_dir
        cfg.MODEL.BACKBONE.FREEZE_AT = 3
        cfg.SOLVER.IMS_PER_BATCH = 5 #BatchSize
        cfg.MODEL.PKDBlock = True
        cfg.MODEL.EDGE = True
        cfg.MODEL.TRAING_STAGE = 'instance'
        cfg.MODEL.MFF = True
        cfg.MODEL.ROI_HEADS.NAME = 'CascadeROIHeads'
        cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
        #cfg.MODEL.DO_FLOPS = True
    if model_name==model_list[1]:
        add_swint_config_tiny(cfg)
        cfg.merge_from_file("./detectron2_code/configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")
        pretrain_dir = "./detectron2_code/pretrain/"
        pretrain_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224_dt2.pth")
        if not os.path.isfile(pretrain_weight_dir):
            orginal_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224.pth")
            if os.path.isfile(orginal_weight_dir):
                swinT_weight_to_dt2(orginal_weight_dir)
            else:
                print('Did not use pretrain weight')
        cfg.MODEL.WEIGHTS = pretrain_weight_dir
        cfg.MODEL.BACKBONE.FREEZE_AT = 3
        cfg.SOLVER.IMS_PER_BATCH = 5 #BatchSize
        #cfg.MODEL.DO_FLOPS = True
    if model_name==model_list[2]:
        add_swint_config_tiny(cfg)
        cfg.merge_from_file("./detectron2_code/configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")
        pretrain_dir = "./detectron2_code/pretrain/"
        pretrain_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224_dt2.pth")
        if not os.path.isfile(pretrain_weight_dir):
            orginal_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224.pth")
            if os.path.isfile(orginal_weight_dir):
                swinT_weight_to_dt2(orginal_weight_dir)
            else:
                print('Did not use pretrain weight')
        cfg.MODEL.WEIGHTS = pretrain_weight_dir
        cfg.MODEL.BACKBONE.FREEZE_AT = 3
        cfg.SOLVER.IMS_PER_BATCH = 5 #BatchSize
        cfg.MODEL.PKDBlock = True
        #cfg.MODEL.DO_FLOPS = True
    if model_name==model_list[3]:
        add_swint_config_tiny(cfg)
        cfg.merge_from_file("./detectron2_code/configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")
        pretrain_dir = "./detectron2_code/pretrain/"
        pretrain_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224_dt2.pth")
        if not os.path.isfile(pretrain_weight_dir):
            orginal_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224.pth")
            if os.path.isfile(orginal_weight_dir):
                swinT_weight_to_dt2(orginal_weight_dir)
            else:
                print('Did not use pretrain weight')
        cfg.MODEL.WEIGHTS = pretrain_weight_dir
        cfg.MODEL.BACKBONE.FREEZE_AT = 3
        cfg.SOLVER.IMS_PER_BATCH = 5 #BatchSize
        cfg.MODEL.PKDBlock = True
        cfg.MODEL.MFF = True
        #cfg.MODEL.DO_FLOPS = True
    if model_name==model_list[4]:
        add_swint_config_tiny(cfg)
        cfg.merge_from_file("./detectron2_code/configs/SwinT/mask_rcnn_swint_T_FPN_3x.yaml")
        pretrain_dir = "./detectron2_code/pretrain/"
        pretrain_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224_dt2.pth")
        if not os.path.isfile(pretrain_weight_dir):
            orginal_weight_dir = os.path.join(pretrain_dir, "swin_tiny_patch4_window7_224.pth")
            if os.path.isfile(orginal_weight_dir):
                swinT_weight_to_dt2(orginal_weight_dir)
            else:
                print('Did not use pretrain weight')
        cfg.MODEL.WEIGHTS = pretrain_weight_dir
        cfg.MODEL.BACKBONE.FREEZE_AT = 3
        cfg.SOLVER.IMS_PER_BATCH = 5 #BatchSize
        cfg.MODEL.PKDBlock = True
        cfg.MODEL.EDGE = True
        cfg.MODEL.TRAING_STAGE = 'instance'
        cfg.MODEL.MFF = True
        #cfg.MODEL.DO_FLOPS = True

    print(cfg.MODEL.RESNETS.OUT_FEATURES)
    print("MODEL_NAME",model_name)
    dataset_list=['cod10k','LVISv1_remove_iscrowd','coco_remove_iscrowd','cod10k_class1']
    dataset_count_list=[3040, 99388,117266,3040]
    dataset_name = 'cod10k_class1'
    if dataset_name==dataset_list[0]:
        cfg.DATASETS.TRAIN = ("cod10k_train",)
        cfg.DATASETS.TEST = ("cod10k_test",) 
        cfg.MODEL.DATASET_COUNT = dataset_count_list[0]
        ITERS_IN_ONE_EPOCH = int(dataset_count_list[0] / cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * epoch) - 1 # epochs，
        solver_checkpoint_period_epoch = 210
        cfg.SOLVER.CHECKPOINT_PERIOD = solver_checkpoint_period_epoch * ITERS_IN_ONE_EPOCH
        cfg.TEST.EVAL_PERIOD_START_ITER = ITERS_IN_ONE_EPOCH * 0 #
        cfg.TEST.EVAL_PERIOD_START_ITER = 1
        cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH * 5
        cfg.SOLVER.BASE_LR = 0.0001 #0.001*4 #學習率越大 batch要越大 預設:16:0.02
        steps = [50,60]
        steps_list = [i * ITERS_IN_ONE_EPOCH for i in steps]
        cfg.SOLVER.STEPS = tuple(steps_list)
        cfg.SOLVER.GAMMA = 0.1      #迭代到指定次數，學習率進行衰減  
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 69 # 類別
        #cfg.MODEL.RPN.NMS_THRESH = 0.7 # RPN proposals上使用的 NMS 閾值 default 0.7
        if 'EDGE' in cfg.MODEL.keys():
            if cfg.MODEL.EDGE:
                #DatasetCatalog.list()
                DatasetCatalog.register('cod10k_edge_train',lambda a='cod10k_train': dataset_insert_edge_data(a))
                cfg.DATASETS.TRAIN = ("cod10k_edge_train",)
                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNNEdge"
                #cfg.INPUT.RANDOM_FLIP = "none"
        cfg.OUTPUT_DIR = "./detectron2_code/result/" + dataset_name + "/" + model_setting_name + "_EPOCH_"+str(epoch)+"/"
    if dataset_name=='LVISv1_remove_iscrowd':
        lr_sched_list = ['lr_sched1x','lr_sched2x']
        lr_sched = lr_sched_list[0]
        if lr_sched == 'lr_sched1x':
            epoch=12            
            steps = [8,11]
        if lr_sched == 'lr_sched2x':             
            epoch=24                         
            steps = [16,22]
        cfg.DATASETS.TRAIN = ("lvis_v1_train",)
        cfg.DATASETS.TEST = ("lvis_v1_val",) 
        #MetadataCatalog.get(dataset_name).set(evaluator_type='lvis')
        ITERS_IN_ONE_EPOCH = int(dataset_count_list[1] / cfg.SOLVER.IMS_PER_BATCH)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1203
        solver_checkpoint_period_epoch = 999
        cfg.SOLVER.CHECKPOINT_PERIOD = solver_checkpoint_period_epoch * ITERS_IN_ONE_EPOCH
        cfg.TEST.EVAL_PERIOD_START_ITER = ITERS_IN_ONE_EPOCH * 1 #
        cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH * 1
        cfg.SOLVER.MAX_ITER = int(round(ITERS_IN_ONE_EPOCH * epoch)) - 1 # epochs，
        cfg.SOLVER.STEPS = tuple([int(round((i * ITERS_IN_ONE_EPOCH))) for i in steps])
        cfg.SOLVER.BASE_LR = 0.0001 #0.001*4 #學習率越大 batch要越大 預設:16:0.02
        cfg.OUTPUT_DIR = "./detectron2_code/result/" + dataset_name + "/" + model_setting_name + "_"+lr_sched+"/"
        cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
        cfg.DATALOADER.REPEAT_THRESHOLD = 0.001
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
        cfg.TEST.DETECTIONS_PER_IMAGE = 300
        if 'EDGE' in cfg.MODEL.keys():             
            if cfg.MODEL.EDGE:
                cfg.DATASETS.TRAIN = ("lvis_v1_Edge_train",)
                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNNEdge"
    if dataset_name=='coco_remove_iscrowd':
        lr_sched_list = ['lr_sched1x','lr_sched3x']
        lr_sched = 'lr_sched1x'
        if lr_sched == 'lr_sched1x':
            epoch=12.28
            steps = [8.19,10.92]
        if lr_sched == 'lr_sched3x':
            epoch=36.84
            steps = [28.65,34.11]
        cfg.DATASETS.TRAIN = ("coco_2017_train",)
        cfg.DATASETS.TEST = ("coco_2017_val",) 
        cfg.MODEL.DATASET_COUNT = dataset_count_list[2]
        ITERS_IN_ONE_EPOCH = int(dataset_count_list[2] / cfg.SOLVER.IMS_PER_BATCH)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        solver_checkpoint_period_epoch = 100
        cfg.SOLVER.CHECKPOINT_PERIOD = solver_checkpoint_period_epoch * ITERS_IN_ONE_EPOCH
        cfg.TEST.EVAL_PERIOD_START_ITER = ITERS_IN_ONE_EPOCH * 1 #
        cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH * 1
        cfg.SOLVER.MAX_ITER = int(round(ITERS_IN_ONE_EPOCH * epoch)) - 1 # epochs，
        cfg.SOLVER.BASE_LR = 0.0001 #0.001*4 #學習率越大 batch要越大 預設:16:0.02
        cfg.SOLVER.STEPS = tuple([int(round((i * ITERS_IN_ONE_EPOCH))) for i in steps])
        cfg.SOLVER.GAMMA = 0.1      #迭代到指定次數，學習率進行衰減  
        cfg.OUTPUT_DIR = "./detectron2_code/result/" + dataset_name + "/" + model_setting_name + "_"+lr_sched+"/"
        if 'EDGE' in cfg.MODEL.keys():             
            if cfg.MODEL.EDGE:
                cfg.DATASETS.TRAIN = ("coco_2017_Edge_train",)
                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNNEdge"
    if dataset_name=='cod10k_class1':
        cfg.DATASETS.TRAIN = ("cod10k_class1_train",)
        cfg.DATASETS.TEST = ("cod10k_class1_test",) 
        cfg.MODEL.DATASET_COUNT = dataset_count_list[3]
        ITERS_IN_ONE_EPOCH = int(dataset_count_list[3] / cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * epoch) - 1 # epochs，
        solver_checkpoint_period_epoch = 210
        cfg.SOLVER.CHECKPOINT_PERIOD = solver_checkpoint_period_epoch * ITERS_IN_ONE_EPOCH
        cfg.TEST.EVAL_PERIOD_START_ITER = ITERS_IN_ONE_EPOCH * 0 #
        cfg.TEST.EVAL_PERIOD_START_ITER = 1
        cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH * 5
        cfg.SOLVER.BASE_LR = 0.0001 #0.001*4 #學習率越大 batch要越大 預設:16:0.02
        steps = [50,60]
        steps_list = [i * ITERS_IN_ONE_EPOCH for i in steps]
        cfg.SOLVER.STEPS = tuple(steps_list)
        
        cfg.SOLVER.GAMMA = 0.1      #迭代到指定次數，學習率進行衰減  
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # 類別
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        #cfg.MODEL.RPN.NMS_THRESH = 0.7 # RPN proposals上使用的 NMS 閾值 default 0.7
        """
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.STEPS = (60000, 80000)
        cfg.SOLVER.MAX_ITER = 90000
        cfg.WEIGHT_DECAY = 0.0001 
        """
        if 'EDGE' in cfg.MODEL.keys():
            if cfg.MODEL.EDGE:
                #DatasetCatalog.list()
                DatasetCatalog.register('cod10k_edge_train',lambda a='cod10k_class1_train': dataset_insert_edge_data(a))
                cfg.DATASETS.TRAIN = ("cod10k_edge_train",)
                cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNNEdge"
                #cfg.INPUT.RANDOM_FLIP = "none"
        cfg.OUTPUT_DIR = "./detectron2_code/result/" + dataset_name + "/" + model_setting_name + "_EPOCH_"+str(epoch)+"/"

    if shut_down_restore_iter != -1:
        cfg.SOLVER.shut_down_restore_iter = shut_down_restore_iter
        cfg.SOLVER.WARMUP_FACTOR = -1         
        cfg.SOLVER.WARMUP_ITERS = -1
        cfg.SOLVER.MAX_ITER = cfg.SOLVER.MAX_ITER-shut_down_restore_iter
        steps = list(cfg.SOLVER.STEPS)
        steps=tuple([i - shut_down_restore_iter for i in steps])
    
    cfg.MODEL.MASK_ON = True
    cfg.DATALOADER.NUM_WORKERS = 0 # 並行數據加載工作者的
    #cfg.OUTPUT_DIR = "./detectron2_code/result/" + dataset_name + "/" + model_setting_name + "_MAX_ITER_"+str(cfg.SOLVER.MAX_ITER)+"/"
    
    

    ####TEST SETTING####
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05# 最低分數閾值（假設分數在 [0, 1] 範圍內）default:0.05
    #cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5#default:0.5
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg
def dataset_insert_edge_data(dataset_name):
    dataset_dict_all = DatasetCatalog.get(dataset_name)
    for i in range(len(dataset_dict_all)):
        dataset_dict_all[i]["Edge_name"] = (dataset_dict_all[i]["file_name"].replace("Image","GT_Edge")).replace("jpg","png")
    return dataset_dict_all
def ISOD_dataset_insert_edge_data(dataset_name):
    dataset_dict_all = DatasetCatalog.get(dataset_name)
    for i in range(len(dataset_dict_all)):
        dataset_dict_all[i]["Edge_name"] = (dataset_dict_all[i]["file_name"].replace("imgs","contour")).replace("jpg","png")
    return dataset_dict_all
def COCO_dataset_insert_edge_data(dataset_name):
    dataset_dict_all = DatasetCatalog.get(dataset_name)
    for i in range(len(dataset_dict_all)):
        dataset_dict_all[i]["Edge_name"] = (dataset_dict_all[i]["file_name"].replace("train2017","contour")).replace("jpg","png")
    return dataset_dict_all
def LVIS_dataset_insert_edge_data(dataset_name):
    dataset_dict_all = DatasetCatalog.get(dataset_name)
    for i in range(len(dataset_dict_all)):
        dataset_dict_all[i]["Edge_name"] = ((dataset_dict_all[i]["file_name"].replace("train2017","contour")).replace("jpg","png")).replace("coco","lvis")
    return dataset_dict_all
def TPVOCIS_dataset_insert_edge_data(dataset_name):
    dataset_dict_all = DatasetCatalog.get(dataset_name)
    for i in range(len(dataset_dict_all)):
        dataset_dict_all[i]["Edge_name"] = (dataset_dict_all[i]["file_name"].replace("train_images","contour_pascal_train")).replace("jpg","png")
    return dataset_dict_all
def TPVOCISv2_dataset_insert_edge_data(dataset_name):
    dataset_dict_all = DatasetCatalog.get(dataset_name)
    for i in range(len(dataset_dict_all)):
        dataset_dict_all[i]["Edge_name"] = (dataset_dict_all[i]["file_name"].replace("train_images","contour")).replace("jpg","png")
    return dataset_dict_all
def setup_dataset():
    register_coco_instances(name='cod10k_train',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Train/CAM_Instance_Train_v2.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Train/Image/')
    register_coco_instances(name='cod10k_test',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Test/CAM_Instance_Test_v2.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Test/Image/')
    register_coco_instances(name='cod10k_class1_train',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Train/train_instance.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Train/Image/')
    register_coco_instances(name='cod10k_class1_test',
                                metadata={},
                                json_file='./detectron2_code/dataset/COD10K-v3/Test/test2026.json',
                                image_root='./detectron2_code/dataset/COD10K-v3/Test/Image/')
    register_coco_instances(name='NC4K',
                                metadata={},
                                json_file='./detectron2_code/dataset/NC4K/test/nc4k_test.json',
                                image_root='./detectron2_code/dataset/NC4K/test/image/')
    register_coco_instances(name='ISOD_train',
                            metadata={},
                            json_file='./detectron2_code/dataset/isod/train.json',
                            image_root='./detectron2_code/dataset/isod/imgs/')
    register_coco_instances(name='ISOD_test',
                            metadata={},
                            json_file='./detectron2_code/dataset/isod/test.json',
                            image_root='./detectron2_code/dataset/isod/imgs/') 
    register_coco_instances(name='TPVOCIS_train',
                            metadata={},
                            json_file='./detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/pascal_train.json',
                            image_root='./detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/train_images/')
    register_coco_instances(name='TPVOCISv2_train',
                            metadata={},
                            json_file='./detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/trainv2.json',
                            image_root='./detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/train_images/')
    register_coco_instances(name='TPVOCISv2_test',
                            metadata={},
                            json_file='./detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/testv2.json',
                            image_root='./detectron2_code/dataset/Tiny Pascal VOC Instance Segmentation/train_images/') 
    DatasetCatalog.register('ISOD_Edge_train',lambda a='ISOD_train': ISOD_dataset_insert_edge_data(a))  
    DatasetCatalog.register('coco_2017_Edge_train',lambda a='coco_2017_train': COCO_dataset_insert_edge_data(a))  
    DatasetCatalog.register('lvis_v1_Edge_train',lambda a='lvis_v1_train': LVIS_dataset_insert_edge_data(a))  
    DatasetCatalog.register('TPVOCIS_edge_train',lambda a='TPVOCIS_train': TPVOCIS_dataset_insert_edge_data(a))  
    DatasetCatalog.register('TPVOCISv2_edge_train',lambda a='TPVOCISv2_train': TPVOCISv2_dataset_insert_edge_data(a))  
def save_setting(cfg):
    filename = cfg.OUTPUT_DIR+'config.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(cfg, f)
def main(args):
    setup_dataset()
    cfg = setup_cfg(args)
    
    os.makedirs(cfg.OUTPUT_DIR,exist_ok=True) 
    save_setting(cfg)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        result_test = do_test(cfg, model)
        return result_test
        #return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    if 'DO_FLOPS' in cfg.MODEL.keys():
        if cfg.MODEL.DO_FLOPS:
            do_flop(cfg,model)
            exit()
    if 'DO_TEST' in cfg.MODEL.keys():
        if cfg.MODEL.DO_TEST:
            do_test(cfg,model)
            exit()
    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', default=False, type=bool, help='eval_only')
    #parser.add_argument('--num-gpus', default=1, type=int, help='gpu')
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )