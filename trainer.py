#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
from typing import Callable, Optional
import faiss

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from utils import progress_bar

import strategies
from custom_datasets import *
from cls_loss import CB_loss
import sampler
import trainer_DP
import train_model

parser = argparse.ArgumentParser(description="Unsupervised distillation")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--dataset", default="imagenet", type=str, help="name of the dataset."
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument("-a", "--arch", default="resnet50", help="model architecture")
parser.add_argument(
    "--epochs", default=40, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--weights",
    dest="weights",
    type=str,
    required=True,
    help="pre-trained model weights",
)
parser.add_argument(
    "--load_cache",
    action="store_true",
    help="should the features be recomputed or loaded from the cache",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--resume-indices",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest selected indices (default: none)",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--save", default="./output", type=str, help="experiment output directory"
)
parser.add_argument(
    "--indices", default="/root/autodl-tmp/pycode/indices", type=str, help="experiment input directory"
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--lr_schedule", type=str, default="30,60,90", help="lr drop schedule"
)
parser.add_argument(
    "--splits", type=str, default="", help="splits of unlabeled data to be labeled"
)
parser.add_argument(
    "--name", type=str, default="", help="name of method to do linear evaluation on."
)
parser.add_argument("--budget", type=float, default=400, help="budget of query")
parser.add_argument(
    "--backbone",
    type=str,
    default="compress",
    help="name of method to do linear evaluation on.",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.99,
    help="beta",
)


def main():
    args = parser.parse_args()
    if not os.path.exists(args.indices):
        os.makedirs(args.indices)

    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_classes = 1000

    elif args.dataset == "imagenet_lt":
        args.num_images = 115846
        args.num_classes = 1000

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100

    elif args.dataset == "cifar10":
        args.num_images = 50000
        args.num_classes = 10

    else:
        raise NotImplementedError
    init_args = copy.deepcopy(args)
    init_args.name = "random"
    init_args.resume = ""

    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    inference_feats, inference_labels = get_inference(args)
    val_loader = trainer_DP.get_val_loader(args.dataset, args)
    init_module(init_args, val_loader, inference_feats, inference_labels)
    main_work(args, val_loader, inference_feats, inference_labels)


# 随机取样初始训练
def init_module(init_args, val_loader, inference_feats, inference_labels):
    print("init module start......")
    # 先random选出初始训练集
    sampler_fc(init_args, inference_feats, inference_labels)
    # 初始训练
    train_fc(init_args, val_loader)


def get_inference(args):
    all_indices = np.arange(args.num_images)

    inference_loader = sampler.get_inference_loader(args.dataset, all_indices, args)
    # backbone = sampler.get_model(args.arch, args.weights, args)
    backbone = train_model.get_backbone_model(args)
    # backbone = train_model.ResNet18()
    backbone = nn.DataParallel(backbone).cuda()
    # backbone.model = nn.DataParallel(backbone.model).cuda()

    cudnn.benchmark = True

    # get all dataset features and labels in eval mode
    inference_feats, inference_labels = sampler.get_feats(
        args.dataset, inference_loader, backbone, args
    )
    return inference_feats, inference_labels


def sampler_fc(args, inference_feats, inference_labels):
    all_indices = np.arange(args.num_images)
    #
    inference_loader = sampler.get_inference_loader(args.dataset, all_indices, args)
    # backbone = sampler.get_model(args.arch, args.weights, args)
    backbone = train_model.get_backbone_model(args)
    # backbone = train_model.ResNet18()
    backbone = nn.DataParallel(backbone).cuda()
    # backbone.model = nn.DataParallel(backbone.model).cuda()
    #
    cudnn.benchmark = True
    #
    # # get all dataset features and labels in eval mode
    inference_feats, inference_labels = sampler.get_feats(
        args.dataset, inference_loader, backbone, args
    )
    # inference_feats, inference_labels = sampler.get_feats(
    #     args.dataset, inference_loader, backbone.model, args
    # )

    split = [int(x) for x in args.splits.split(",")]

    current_indices = None
    save_indices = None
    if os.path.isfile(args.resume_indices):
        print("=> Loading current indices: {}".format(args.resume_indices))
        current_indices = np.load(args.resume_indices)
        save_indices = np.load(args.resume_indices)
        print(
            "current indices size: {}. {}% of all categories is seen".format(
                len(current_indices),
                len(np.unique(inference_labels[current_indices]))
                / args.num_classes
                * 100,
            )
        )
    # if os.path.isfile('/home/wulongdog/pycode/indices/{}_{}_{}.npy'.format(args.name,args.dataset,split)):
    #     print("=> Loading current indices: /home/wulongdog/pycode/indices/{}_{}_{}.npy".format(args.name,args.dataset,split))
    #     current_indices = np.load('/home/wulongdog/pycode/indices/{}_{}_{}.npy'.format(args.name,args.dataset,split))
    #     print('current indices size: {}. {}% of all categories is seen'.format(len(current_indices), len(np.unique(inference_labels[current_indices])) / args.num_classes * 100))
    elif os.path.isfile(
        "/root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(args.dataset, split[0])
    ):
        print(
            "=> Loading current indices: /root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(
                args.dataset, split[0]
            )
        )
        current_indices = np.load(
            "/root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(
                args.dataset, split[0]
            )
        )
        save_indices = np.load(
            "/root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(
                args.dataset, split[0]
            )
        )
        print(
            "current indices size: {}. {}% of all categories is seen".format(
                len(current_indices),
                len(np.unique(inference_labels[current_indices]))
                / args.num_classes
                * 100,
            )
        )

    if args.name == "uniform":
        print(f"Query sampling with {args.name} started ...")
        strategies.uniform(inference_labels, split, args)
        return

    if args.name == "random":
        print(f"Query sampling with {args.name} started ...")
        strategies.random(all_indices, inference_labels, split, args)
        return

    unlabeled_indices = np.setdiff1d(all_indices, save_indices)
    print(f"Current unlabeled indices is {len(unlabeled_indices)}.")

    if args.name == "kmeans":
        print(f"Query sampling with {args.name} started ...")
        current_indices = strategies.fast_kmeans(
            inference_feats, split[0], args
        )

    elif args.name == "accu_kmeans":
        print(f"Query sampling with {args.name} started ...")
        sampled_indices = strategies.accu_kmeans(
            inference_feats, split[0], unlabeled_indices, args
        )
        current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)

    elif args.name == "coreset":
        print(f"Query sampling with {args.name} started ...")
        sampled_indices = strategies.core_set(
            inference_feats[unlabeled_indices],
            inference_feats[current_indices],
            unlabeled_indices,
            split[0],
            args,
        )
        current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)

    else:
        raise NotImplementedError("Query sampling method is not implemented")

    print(
        "{} inidices are sampled in total, {} of them are unique".format(
            len(current_indices), len(np.unique(current_indices))
        )
    )
    print(
        "{}% of all categories is seen".format(
            len(np.unique(inference_labels[current_indices])) / args.num_classes * 100
        )
    )
    save_indices = np.append(save_indices, current_indices)
    print(
        "{}/{}_{}_{}.npy".format(
            args.indices, args.name, args.dataset, len(save_indices)
        )
    )
    np.save(
        f"{args.indices}/{args.name}_{args.dataset}_{len(save_indices)}.npy",
        save_indices,
    )


def train_fc(args, val_loader):
    # 加载模型
    print("=> creating model")
    # task_model = models.__dict__[args.arch](num_classes=args.num_classes)
    task_model = train_model.get_backbone_model(args)
    # task_model = train_model.ResNet18()
    task_model = torch.nn.DataParallel(task_model).cuda()
    # task_model.model = torch.nn.DataParallel(task_model.model).cuda()
    # task_model.classifier = torch.nn.DataParallel(task_model.classifier).cuda()

    # optimizer = torch.optim.SGD(
    #     task_model.model.parameters(),
    #     args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )

    optimizer = torch.optim.SGD(
        task_model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    sched = [int(x) for x in args.lr_schedule.split(",")]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched)
    # validation data loading code
    # val_loader = trainer_DP.get_val_loader(args.dataset, args)

    split = [int(x) for x in args.splits.split(",")]

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume)

            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            acc1 = checkpoint["best_acc1"]
            # task_model.model.load_state_dict(checkpoint["state_dict"])
            # task_model.classifier.load_state_dict(checkpoint["classifier"])
            task_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        elif os.path.isfile(
            "/root/autodl-tmp/pycode/random_{}_{}.pth.tar".format(args.dataset, split[0])
        ):
            print(
                "=> loading checkpoint '/root/autodl-tmp/pycode/random_{}_{}.pth.tar'".format(
                    args.dataset, split[0]
                )
            )

            checkpoint = torch.load(
                "/root/autodl-tmp/pycode/random_{}_{}.pth.tar".format(
                    args.dataset, split[0]
                )
            )

            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            acc1 = checkpoint["best_acc1"]
            # task_model.model.load_state_dict(checkpoint["state_dict"])
            # task_model.classifier.load_state_dict(checkpoint["classifier"])
            task_model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_acc1 = 0

    cudnn.benchmark = True

    current_indices = []
    # current indices loading
    curr_idxs_file = "{}/{}_{}_{}.npy".format(
        args.indices, args.name, args.dataset, split[-1]
    )
    if os.path.isfile(curr_idxs_file):
        print("=> Loading current indices: {}".format(curr_idxs_file))
        current_indices = np.load(curr_idxs_file)
        print("current indices size: {}.".format(len(current_indices)))
    elif os.path.isfile("/root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(args.dataset, split[0])):
        print(
            "=> Loading current indices: /root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(
                args.dataset, split[0]
            )
        )
        current_indices = np.load(
            "/root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(args.dataset, split[0])
        )
        print("current indices size: {}.".format(len(current_indices)))
    else:
        print("=> no such file found at '{}'".format(curr_idxs_file))
        print(
            "=> no such file found at '/root/autodl-tmp/pycode/indices/random_{}_{}.npy".format(
                args.dataset, split[0]
            )
        )

    # Training data loading code
    train_loader = trainer_DP.get_train_loader(args.dataset, current_indices, args)

    print("Training task model started ...")
    args.start_epoch = 0
    print("start epoch is{} and epoch is {}".format(args.start_epoch, args.epochs))
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, task_model, optimizer, epoch, args)
        # train(epoch, task_model, train_loader, optimizer)
        # evaluate on validation set
        # if epoch % 10 == 0 or epoch == args.epochs - 1:
        acc1 = validate(task_model, val_loader, args)
            # acc1 = test(epoch, task_model, val_loader, optimizer)
            # remember best acc@1 and save checkpoint
            
        if acc1 > best_acc1:
            best_acc1 = max(acc1, best_acc1)

            trainer_DP.save_checkpoint(
                {
                    "epoch": epoch + 1,
                        # "state_dict": task_model.model.state_dict(),
                    "state_dict": task_model.state_dict(),
                        # "classifier": task_model.classifier.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                "{}_{}_{}.pth.tar".format(args.name, args.dataset, split[-1]),
            )

        # lr_scheduler.step()
        print("LR: {:f}".format(lr_scheduler.get_last_lr()[-1]))

    print("Final accuracy of {} labeled data is: {:.2f}".format(split[-1], best_acc1))


def train(train_loader, task_model, optimizer, epoch, args):
    batch_time = trainer_DP.AverageMeter('Time', ':6.3f')
    data_time = trainer_DP.AverageMeter('Data', ':6.3f')
    losses = trainer_DP.AverageMeter('Loss', ':.4e')
    top1 = trainer_DP.AverageMeter('Acc@1', ':6.2f')
    top5 = trainer_DP.AverageMeter('Acc@5', ':6.2f')
    progress = trainer_DP.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    task_model.train()

    end = time.time()

    for i, (images, target, _) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        temp = torch.unique(target, return_counts=True)
        target_list = []
        index = 0
        for j in range(10):
            if len(temp[0]) > index or temp[0][index] != j:
                target_list.append(0)
            else:
                target_list.append(temp[1][index])
                index = index + 1

        samples_per_cls = torch.tensor(target_list)

        samples_per_cls = samples_per_cls.data.cpu()

        # compute output
        output = task_model(images)
        # loss = F.cross_entropy(output, target)
        # loss = CB_loss(target,output,samples_per_cls,len(samples_per_cls),'softmax',args.beta,1)
        # loss = CB_loss(target,output,samples_per_cls,len(samples_per_cls),'softmax',args.beta,1)
        effective_num = 1.0 - np.power(0.99, samples_per_cls)
        effective_num = np.array(effective_num)
        effective_num = np.maximum(effective_num, np.finfo(np.float32).eps)
        weights = (1.0 - 0.99) / effective_num
        weights = weights / np.sum(weights) * 10

        # loss = F.cross_entropy(output,target,torch.tensor(weights,dtype=torch.float).cuda(non_blocking=True))

        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        acc1, acc5 = trainer_DP.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(task_model, val_loader, args):
    batch_time = trainer_DP.AverageMeter('Time', ':6.3f')
    losses = trainer_DP.AverageMeter('Loss', ':.4e')
    top1 = trainer_DP.AverageMeter('Acc@1', ':6.2f')
    top5 = trainer_DP.AverageMeter('Acc@5', ':6.2f')
    progress = trainer_DP.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    task_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            temp = torch.unique(target, return_counts=True)
            target_list = []
            index = 0
            for j in range(10):
                if len(temp[0]) > index or temp[0][index] != j:
                    target_list.append(0)
                else:
                    target_list.append(temp[1][index])
                    index = index + 1

            samples_per_cls = torch.tensor(target_list)

            samples_per_cls = samples_per_cls.data.cpu()
            # compute output
            output = task_model(images)
            # loss = F.cross_entropy(output, target)
            # loss = CB_loss(target,output,samples_per_cls,len(samples_per_cls),'softmax',args.beta,1)
            effective_num = 1.0 - np.power(0.99, samples_per_cls)
            effective_num = np.array(effective_num)
            effective_num = np.maximum(effective_num, np.finfo(np.float32).eps)
            weights = (1.0 - 0.99) / effective_num
            weights = weights / np.sum(weights) * 10

            # loss = F.cross_entropy(output,target,torch.tensor(weights,dtype=torch.float).cuda(non_blocking=True))
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = trainer_DP.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def main_work(args, val_loader, inference_feats, inference_labels):
    print("start main train......")
    i = 1
    split = int(args.splits)
    while args.budget >= split:
        print("{} train action......".format(i))
        args.budget -= int([x for x in args.splits.split(",")][0])
        args.splits += ",{}".format(split + 100 * i)
        # 挑选样本
        args.resume_indices = "{}/{}_{}_{}.npy".format(
            args.indices, args.name, args.dataset, split + 100 * (i-1)
        )
        sampler_fc(args, inference_feats, inference_labels)
        
        args.resume = "/root/autodl-tmp/pycode/{}_{}_{}.pth.tar".format(
            args.name, args.dataset, split + 100 * (i-1)
        )
        # 训练模型
        train_fc(args, val_loader)
        args.num_images = 50000
        i += 1
    print("have weights")


if __name__ == "__main__":
    main()
