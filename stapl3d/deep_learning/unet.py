#!/usr/bin/env python

"""Torch functionality for STAPL3D.

Model development by Sam de Blank.
Adapted for STAPL3D by Michiel Kleinnnijenhuis.

# https://www.arxiv-vanity.com/papers/1907.09140/
After obtaining the bounding boxes for all cell instances in the input images, 
we perform the individual cell segmentation for each cell instance. 
Motivated by U-net [10], we combine the feature maps from the shallow layers 
with the feature maps from the deep layers to take advantage of both high-level 
semantics and low-level image details. Specifically, we crop the multi-scale 
feature maps from the backbone network (see Fig. 1b) and then perform a 
bottom-up segmentation for the cropped cell patchs. Note that we intentionally 
employ an individual cell segmentation branch (Fig. 1b) for cell segmentation 
instead of directly reusing the feature map at s1 (Fig. 1a). Our motivation is 
to use the branch to guide the model to eliminate the interference from 
neighboring cells and learn an objectness concept especially for cells with 
irregular shapes (see Fig. 3).
"""

import os
import sys
import logging
import pickle
import shutil
import multiprocessing
import random
import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from stapl3d import parse_args
from stapl3d.blocks import Stapl3r, Block3r

from stapl3d.deep_learning import models, transforms, datasets, checkpoint

logger = logging.getLogger(__name__)


def main(argv):
    """."""

    steps = ['train', 'predict']
    args = parse_args('torch', steps, *argv)

    torch3r = Torch3r(
        args.image_in,
        args.parameter_file,
        step_id=args.step_id,
        directory=args.outputdir,
        prefix=args.prefix,
        max_workers=args.max_workers,
    )

    for step in args.steps:
        torch3r._fun_selector[step]()


class Torch3r(Block3r):
    _doc_main = """Pytorch in stapl3d."""
    _doc_attr = """

    Torch3r Attributes
    ----------
    """
    _doc_meth = """

    Torch3r Methods
    --------
    run
        Run all steps in the Torch3r module.
    train
        ...
    predict
        ...
    view
        View volumes with napari.
    """
    _doc_exam = """

    Torch3r Examples
    --------
    # TODO
    """
    __doc__ = f"{_doc_main}{Stapl3r.__doc__}{_doc_meth}{_doc_attr}{_doc_exam}"

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'torch'

        super(Torch3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector.update({
            'train': self.train,
            'predict': self.predict,
            })

        self._parallelization.update({
            'train': ['blocks'],
            'predict': ['blocks'],
            })

        self._parameter_sets.update({
            'train': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize', 'blockmargin'),
                'spar': ('_n_workers',),
                },
            'predict': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('blocksize', 'blockmargin'),
                'spar': ('_n_workers', 'blocks'),
                },
            })

        self._parameter_table.update({
            })

        default_attr = {
            'ids_image': 'memb/mean',
            'ids_labels': 'memb/mask',
            'ods_image': 'memb/prob',
            'ods_labels': '',
            'augment': [],
            'epochs': 2,
            'test_every_n_epochs': 1,
            'checkpoint_every_n_epochs': None,
            'learning_rate': 0.0001,
            'n_kernels': [64, 128, 256, 512],
            'n_classes': 1,
            'class_weights': None,
            'train_indices': [],
            'val_indices': [],
            'val_fraction': 0.15,
            'save_images': False,
            'metrics': ["dice", "accuracy"],
            'load_model': '',
        }

        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step, step_id=step_id)

        self._init_paths_torcher()

        self._init_log()

        self._prep_blocks()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.checkpoint_every_n_epochs is None:
            self.checkpoint_every_n_epochs = self.test_every_n_epochs

        # axlab = self._blocks[0].axlab
        # self._axlab = ''.join([al for al in axlab if al in 'zyx'])

        self._images = []
        self._labels = []

    def _init_paths_torcher(self):

        blockfiles = self.outputpaths['blockinfo']['blockfiles']
        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        stem = self._build_path(
            moduledir='blocks',
            prefixes=[self.prefix, 'blocks'],
            )

        self._paths.update({
            'train': {
                'inputs': {
                    'blockfiles': blockfiles,
                    'model_path': '',
                    },
                'outputs': {
                    'blockfiles': blockfiles,
                    'model_path': 'my_checkpoint.pth.tar',
                    'saved_images': '.',
                    }
                },
            'predict': {
                'inputs': {
                    'blockfiles': blockfiles,
                    'model_path': 'my_checkpoint.pth.tar',
                    },
                'outputs': {
                    'blockfiles': blockfiles,
                    }
                },
            })

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)

    def train(self):

        step = 'train'
        _ = self._prep_step(step)

        self._train()

    def _train(self):

        inputs = self._prep_paths(self.inputs)
        outputs = self._prep_paths(self.outputs)

        log_dir = os.path.join(self._logdir, 'tb')
        tensorboard_writer = SummaryWriter(log_dir=log_dir)
        checkpoint_handler = checkpoint.CheckpointHandler()

        # TODO: option to load from external file
        # model, loss, optimizer, scalar
        model = models.UNET(
            in_channels=1,
            out_channels=self.n_classes,
            features=self.n_kernels,
            ).to(self.device)
        if self.device.startswith("cuda") and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()

        if self.n_classes == 1:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        scaler = torch.cuda.amp.GradScaler()

        if not self.val_indices:
            indices = list(range(len(self._blocks)))
            random.shuffle(indices)
            N = int((1 - self.val_fraction) * len(indices))
            self.train_indices = indices[:N]
            self.val_indices = indices[N:]
            # TODO: save?

        train_blocks = [self._blocks[idx] for idx in self.train_indices]
        train_loader = self.get_loader(train_blocks, augmenter=self.augment)
        val_blocks = [self._blocks[idx] for idx in self.val_indices]
        val_loader = self.get_loader(val_blocks, augmenter=None)

        epoch_start = 1
        if inputs['model_path']:
            pass

        for epoch in range(epoch_start, self.epochs + 1):

            print(f"\n### Epoch: {epoch}")

            train_metrics = self.train_fn(
                blocks=train_blocks if self.save_images else None,
                loader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                )
            store_metrics('train', epoch, train_metrics, tensorboard_writer, checkpoint_handler)

            if epoch % self.test_every_n_epochs == 0:

                test_metrics, images = self.test_fn(
                    blocks=val_blocks if self.save_images else None,
                    loader=val_loader,
                    model=model,
                    loss_fn=loss_fn,
                )

                store_metrics('test', epoch, test_metrics, tensorboard_writer, checkpoint_handler)
                store_images('val_images', epoch, images, tensorboard_writer, checkpoint_handler)

            if epoch % self.checkpoint_every_n_epochs == 0:

                print("Creating checkpoint...")
                checkpoint_handler.save_checkpoint(
                    checkpoint_path=outputs['model_path'],
                    iteration=epoch,
                    model=model,
                    optimizer=optimizer,
                    )

    def train_fn(self, blocks, loader, model, loss_fn, optimizer, scaler):
        """"""

        loop = tqdm(loader)

        model.train()

        for data, targets, index in loop:

            data = data.to(device=self.device)
            targets = targets.to(device=self.device)

            # forward
            with torch.cuda.amp.autocast():
                preds = model(data)
                loss = loss_fn(preds, targets)
            # backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=float(loss.item()))

        train_metrics = {"Loss": float(loss)}

        if self.n_classes > 1:
            preds_thr, _, _ = multiclass_predict(preds)
            add_metrics, _ = calculate_multiclass_scores(
                preds, targets,
                nr_classes=self.n_classes,
                metrics=self.metrics,
                preds_thr=preds_thr,
                )
        else:
            preds_thr, _ = binary_predict(preds)
            add_metrics, _ = calculate_binary_scores(
                preds, targets,
                metrics=self.metrics,
                preds_thr=preds_thr,
                )

        train_metrics.update(add_metrics)

        return train_metrics

    def test_fn(self, blocks, loader, model, loss_fn):
        """"""

        running_loss = 0
        running_calcs = {}

        model.eval()

        with torch.no_grad():

            images = {}
            for data, targets, index in loader:

                data = data.to(self.device)
                targets = targets.to(self.device)

                with torch.cuda.amp.autocast():
                    preds = model(data)
                    loss = loss_fn(preds, targets)

                if self.n_classes > 1:
                    preds_thr, preds_prob, _ = multiclass_predict(preds)
                    _, batch_calcs = calculate_multiclass_scores(
                        preds, targets,
                        nr_classes=self.n_classes,
                        metrics=self.metrics,
                        )
                else:
                    preds_thr, preds_prob = binary_predict(preds)
                    _, batch_calcs = calculate_binary_scores(
                        preds, targets,
                        metrics=self.metrics,
                        )

                running_calcs = {k: running_calcs.get(k, 0) + batch_calcs.get(k, 0)
                                 for k in set(batch_calcs) | set(running_calcs)}

                running_loss += loss

                centreslices = self.save_results(
                    blocks[index] if self.save_images else None,
                    data=data,
                    preds_thr=preds_thr,
                    preds_prob=preds_prob,
                    targets=targets,
                )

                images[index.item()] = centreslices

        test_metrics = {"Loss": float(running_loss / len(loader))}
        if self.n_classes > 1:
            add_metrics, _ = calculate_multiclass_scores(
                preds, targets,
                nr_classes=self.n_classes,
                metrics=self.metrics,
                calcs=running_calcs,
                )
        else:
            add_metrics, _ = calculate_binary_scores(
                preds, targets,
                metrics=self.metrics,
                calcs=running_calcs,
                )

        test_metrics.update(add_metrics)

        return test_metrics, images

    def get_loader(self, blocks, augmenter=None,
                   batch_size=1, n_workers=4, pin_memory=True):

        def get_A(aug):
            for aug_name, aug_kwargs in aug.items():
                return eval(f'transforms.{aug_name}')(**aug_kwargs)

        if augmenter is not None:
            augmenter = transforms.Augmenter([get_A(aug) for aug in augmenter])

        ds = datasets.Dataset_STAPL3D(blocks, self.ids_image, self.ids_labels, augmenter)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=pin_memory,
            shuffle=True
        )

        return loader

    def save_results(self, block, data, preds_thr, preds_prob=None, targets=None):

        def update_cslices(cslices, slc, N, cols, al=''):
            for c in cols:
                if not al:
                    cslices[N, c, :, :] = slc
                elif al == 'z':
                    cslices[N, c, :slc.shape[0] , :slc.shape[1] ] = slc
                elif al == 'y':
                    cslices[N, c, -slc.shape[0]:, :slc.shape[1] ] = slc
                elif al == 'x':
                    cslices[N, c, :slc.shape[1] , -slc.shape[0]:] = slc.T

            return cslices

        def write_output(block, ods, out, imtype):
            block.create_dataset(
                ods,
                axlab='zyx',
                dtype=out.dtype,
                imtype=imtype,
                create_image=True,
                )
            block.datasets[ods].write(out)

        data = data.cpu().numpy().squeeze()
        targets = targets.cpu().numpy().squeeze()
        preds_prob = preds_prob.cpu().numpy().squeeze()
        preds_thr = preds_thr.cpu().numpy().squeeze()  # TODO: convert to bool/int

        if block is not None:
            if self.ods_image:
                write_output(block, self.ods_image, preds_prob, '')
            if self.ods_labels:
                write_output(block, self.ods_labels, preds_thr, 'Mask')




        c_slcs = {}

        axislabels = 'zyx'
        axes = dict(zip(axislabels, list(range(len(axislabels)))))
        cslc_shape = [data.shape[axes['x']] + 1 + data.shape[axes['z']],
                      data.shape[axes['y']] + 1 + data.shape[axes['z']]]
        cslices = np.zeros([4, 3] + cslc_shape)

        im_dicts = {
            0: {'cols': [0, 1, 2], 'data': data},
            1: {'cols': [0, 1, 2], 'data': preds_prob},
            2: {'cols': [0], 'data': targets},
            3: {'cols': [1], 'data': preds_thr},
        }

        for al, axis in axes.items():

            idx = int(data.shape[axis] / 2)
            slcs = [slice(None)] * 3
            slcs[axis] = slice(idx, idx + 1)

            for N, v in im_dicts.items():
                cslices = update_cslices(
                    cslices,
                    np.squeeze(v['data'][tuple(slcs)]),
                    N,
                    v['cols'],
                    al,
                    )

        cslices[:, 0, data.shape[axes['x']]+1, :] = 1  # yellow line ortho separator
        cslices[:, 1, data.shape[axes['x']]+1, :] = 1  # yellow line ortho separator
        cslices[:, 0, :, data.shape[axes['y']]+1] = 1  # yellow line ortho separator
        cslices[:, 1, :, data.shape[axes['y']]+1] = 1  # yellow line ortho separator

        c_slcs['ortho'] = cslices

        return c_slcs

    def predict(self):
        pass


def store_metrics(stage, epoch, metrics, tensorboard_writer, checkpoint_handler):

    for k, v in metrics.items():

        print(f"{k}: {v:.3f}")

        tensorboard_writer.add_scalars(
            k,
            {stage : metrics[k]},
            epoch,
        )

        checkpoint_handler.store_running_var_with_header(
            header=stage,
            var_name=k,
            iteration=epoch,
            value=metrics[k],
            )


def store_images(stage, epoch, image_pairs, tensorboard_writer, checkpoint_handler):

    for val_idx, image_dict in image_pairs.items():
        for name, image_set in image_dict.items():  # e.g. ortho-projection
            # image_set is 4D NCHW: N-> data - prob - mask - P>0.5
            tensorboard_writer.add_images(
                f'{stage}_{name}_{val_idx}',
                image_set,
                epoch,
                dataformats='NCHW',
            )


def binary_predict(preds):
    """"""

    preds_sigmoid = torch.sigmoid(preds)
    preds_thr = (preds_sigmoid > 0.5).float()

    return preds_thr, preds_sigmoid


def calculate_binary_scores(preds, targets, metrics=["dice", "accuracy"], calcs=None, preds_thr=None):
    """"""

    if preds_thr == None:
        preds_thr, _= binary_predict(preds)

    all_metrics = {}

    if calcs == None:
        calcs = {}

    for metric in metrics:
        if metric.lower() == "accuracy":
            if "nr_correct" not in calcs or "nr_images" not in calcs:
                calcs["nr_correct"] = float((preds_thr==targets).sum())
                calcs["nr_images"] = int(torch.numel(preds_thr))
            all_metrics["Accuracy"] = float(calcs["nr_correct"] / calcs["nr_images"])

    return all_metrics, calcs


def multiclass_predict(preds):
    """"""

    preds_softmax = torch.nn.functional.softmax(preds, dim=1)
    max_idx = torch.argmax(preds, dim=1, keepdim=True)
    preds_thr = torch.empty(preds.shape, device=preds.device)
    preds_thr.zero_()
    preds_thr.scatter_(1, max_idx, 1)

    return preds_thr, preds_softmax, max_idx


def calculate_multiclass_scores(preds, targets, nr_classes=1, metrics=["dice", "accuracy"], calcs=None, preds_thr=None):
    """"""

    all_metrics = {}

    if calcs==None:
        calcs={}

    if preds_thr==None:
        preds_thr, _, _ = multiclass_predict(preds)

    for img_class in range(nr_classes):
        class_preds_thr = torch.index_select(preds_thr, 1, torch.tensor([img_class], device=preds_thr.device))
        class_targets = (targets==img_class).type(torch.uint8).unsqueeze(1)

        for metric in metrics:
            if metric.lower()=="dice":
                if f"class{img_class}_dice_num" not in calcs and f"class{img_class}_dice_denom" not in calcs:
                    calcs[f"class{img_class}_dice_num"] = 2 * float(( class_preds_thr * class_targets ).sum())
                    calcs[f"class{img_class}_dice_denom"] = float((class_preds_thr + class_targets).sum())
                all_metrics[f"class{img_class}_dice"] = float(calcs[f"class{img_class}_dice_num"] / ( calcs[f"class{img_class}_dice_denom"] + 1e-8 ))
            if metric.lower()=="accuracy":
                if f"class{img_class}_nr_correct" not in calcs or f"class{img_class}_nr_pixels" not in calcs:
                    calcs[f"class{img_class}_nr_correct"] = float((class_preds_thr==class_targets).sum())
                    calcs[f"class{img_class}_nr_pixels"] = int(torch.numel(class_preds_thr))
                all_metrics[f"class{img_class}_accuracy"] = float(calcs[f"class{img_class}_nr_correct"] / calcs[f"class{img_class}_nr_pixels"])

    return all_metrics, calcs


if __name__ == "__main__":
    main(sys.argv[1:])
