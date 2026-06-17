# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

import numpy as np
import time
import paddle
import paddle.nn.functional as F

from paddleseg.utils import metrics, TimeAverager, calculate_eta, logger, progbar
from paddleseg.core import infer

np.set_printoptions(suppress=True)


def _resolve_class_names(num_classes, class_names=None):
    if class_names is None:
        return ["class_{}".format(i) for i in range(num_classes)]

    if isinstance(class_names, str):
        if os.path.exists(class_names):
            with open(class_names, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
        else:
            class_names = [name.strip() for name in class_names.split(',') if name.strip()]
    elif len(class_names) == 1 and os.path.exists(class_names[0]):
        with open(class_names[0], 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]

    if len(class_names) != num_classes:
        logger.warning(
            "The number of class names ({}) does not match num_classes ({}). "
            "Fallback to class indexes.".format(len(class_names), num_classes))
        return ["class_{}".format(i) for i in range(num_classes)]

    return list(class_names)


def _format_class_metrics_table(class_iou,
                                class_f1,
                                class_acc,
                                intersect_area,
                                pred_area,
                                label_area,
                                class_names=None):
    num_classes = len(class_iou)
    names = _resolve_class_names(num_classes, class_names)
    rows = []
    for idx in range(num_classes):
        rows.append([
            str(idx),
            names[idx],
            "{:.4f}".format(float(class_iou[idx])),
            "{:.4f}".format(float(class_f1[idx])),
            "{:.4f}".format(float(class_acc[idx])),
            str(int(intersect_area[idx])),
            str(int(pred_area[idx])),
            str(int(label_area[idx])),
        ])

    headers = ["Class", "Name", "IoU", "F1", "Acc", "Intersect", "Pred", "Label"]
    widths = [
        max(len(headers[col]), max(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    header_line = " | ".join(headers[col].ljust(widths[col]) for col in range(len(headers)))
    sep_line = "-+-".join("-" * width for width in widths)
    row_lines = [
        " | ".join(row[col].ljust(widths[col]) for col in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def evaluate(model,
             eval_dataset,
             aug_eval=False,
             swap=False,
             scales=1.0,
             batch_size=1,
             flip_horizontal=True,
             flip_vertical=False,
             is_slide=False,
             stride=None,
             crop_size=None,
             num_workers=0,
             print_detail=True,
             class_table=False,
             class_names=None):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A sementic segmentation model.
        eval_dataset (paddle.io.Dataset): Used to read and process validation datasets.
        aug_eval (bool, optional): Whether to use mulit-scales and flip augment for evaluation. Default: False.
        scales (list|float, optional): Scales for augment. It is valid when `aug_eval` is True. Default: 1.0.
        flip_horizontal (bool, optional): Whether to use flip horizontally augment. It is valid when `aug_eval` is True. Default: True.
        flip_vertical (bool, optional): Whether to use flip vertically augment. It is valid when `aug_eval` is True. Default: False.
        is_slide (bool, optional): Whether to evaluate by sliding window. Default: False.
        stride (tuple|list, optional): The stride of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        crop_size (tuple|list, optional):  The crop size of sliding window, the first is width and the second is height.
            It should be provided when `is_slide` is True.
        num_workers (int, optional): Num workers for data loader. Default: 0.
        print_detail (bool, optional): Whether to print detailed information about the evaluation process. Default: True.
        class_table (bool, optional): Whether to print per-class metrics as a table. Default: False.
        class_names (list[str]|str, optional): Class names or a class-name file. Default: None.

    Returns:
        float: The mIoU of validation datasets.
        float: The accuracy of validation datasets.
    """
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(
        p.numel() for p in model.parameters() if not p.stop_gradient)

    nranks = paddle.distributed.ParallelEnv().nranks
    local_rank = paddle.distributed.ParallelEnv().local_rank
    if nranks > 1:
        # Initialize parallel environment if not done.
        if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
        ):
            paddle.distributed.init_parallel_env()
    batch_sampler = paddle.io.DistributedBatchSampler(
        eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    loader = paddle.io.DataLoader(
        eval_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        return_list=True,
    )

    total_iters = len(loader)
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    if print_detail:
        logger.info(
            "Start evaluating (total_samples: {}, total_iters: {})...".format(
                len(eval_dataset), total_iters))
    #TODO(chenguowei): fix log print error with multi-gpus
    progbar_val = progbar.Progbar(
        target=total_iters, verbose=1 if nranks < 2 else 2)
    reader_cost_averager = TimeAverager()
    batch_cost_averager = TimeAverager()
    total_time = 0.0
    input_shape = None
    batch_start = time.time()
    with paddle.no_grad():
        for iter, data in enumerate(loader):
            im1, im2, label = data
            reader_cost_averager.record(time.time() - batch_start)
            label = label.astype('int64')
            ori_shape = label.shape[-2:]
            if aug_eval:
                pred = infer.aug_inference(
                    model,
                    im1,
                    im2,
                    swap=swap,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    scales=scales,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    is_slide=is_slide,
                    stride=stride,
                    crop_size=crop_size)
            else:
                pred = infer.inference(
                    model,
                    im1,
                    im2,
                    ori_shape=ori_shape,
                    transforms=eval_dataset.transforms.transforms,
                    stride=stride,
                    crop_size=crop_size)

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred,
                label,
                eval_dataset.num_classes,
                ignore_index=eval_dataset.ignore_index)

            if input_shape is None:
                c1, h1, w1 = im1.shape[1], im1.shape[2], im1.shape[3]
                c2, h2, w2 = im2.shape[1], im2.shape[2], im2.shape[3]
                input_shape = (c1, h1, w1, c2, h2, w2)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                paddle.distributed.all_gather(intersect_area_list,
                                              intersect_area)
                paddle.distributed.all_gather(pred_area_list, pred_area)
                paddle.distributed.all_gather(label_area_list, label_area)

                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(eval_dataset):
                    valid = len(eval_dataset) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area
            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(label))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()
            total_time += (batch_cost - reader_cost)

            if local_rank == 0 and print_detail:
                progbar_val.update(iter + 1, [('batch_cost', batch_cost),
                                              ('reader cost', reader_cost)])
            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()


    # Compute FLOPs after all inference to avoid perturbing BN stats
    flops = None
    if input_shape is not None:
        try:
            c1, h1, w1, c2, h2, w2 = input_shape
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='When training, we now always track global mean and variance.'
                )
                from paddleslim.analysis import flops as slim_flops
                flops = slim_flops(model, inputs=[[1, c1, h1, w1], [1, c2, h2, w2]])
        except Exception:
            try:
                class _FlopsWrapper(paddle.nn.Layer):
                    def __init__(self, model, c1, c2):
                        super().__init__()
                        self.model = model
                        self.c1 = c1
                        self.c2 = c2
                    def forward(self, x):
                        return self.model(x[:, :self.c1], x[:, self.c1:])
                wrapper = _FlopsWrapper(model, c1, c2)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        message='When training, we now always track global mean and variance.'
                    )
                    flops = paddle.flops(wrapper, [1, c1 + c2, h1, w1])
            except Exception:
                flops = None

    class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                       label_area_all)
    class_f1, f1  = metrics.get_f1(intersect_area_all, pred_area_all, label_area_all)
    class_acc, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    if print_detail:
        logger.info(
            "[EVAL] #Images: {} F1: {:.4f}, mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} ".format(
                len(eval_dataset), f1, miou, acc, kappa))
        fps = len(eval_dataset) / total_time if total_time > 0 else 0
        flops_str = "{:.2f}G".format(flops / 1e9) if flops is not None else "N/A"
        logger.info(
            "[EVAL] Params: {:.2f}M  Trainable: {:.2f}M  FLOPs: {}  FPS: {:.2f}".format(
                num_params / 1e6, num_trainable / 1e6, flops_str, fps))
        logger.info("[EVAL] Class IoU: \n" + str(np.round(class_iou, 4)))
        logger.info("[EVAL] Class Acc: \n" + str(np.round(class_acc, 4)))
        logger.info("[EVAL] Class F1: \n" + str(np.round(class_f1, 4)))
        if class_table:
            logger.info("[EVAL] Per-class metrics:\n" + _format_class_metrics_table(
                class_iou,
                class_f1,
                class_acc,
                intersect_area_all.numpy(),
                pred_area_all.numpy(),
                label_area_all.numpy(),
                class_names))
    return miou, acc, class_iou, class_acc, kappa
