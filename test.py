from email.mime import image
from random import shuffle
import numpy as np
import os
import argparse
import torch
import faiss
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
from cls_loss import CB_loss, cb_
from train_model import get_my_backbone_model
from trainer_DP import accuracy
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf

from custom_datasets import CIFAR10Subset

parser = argparse.ArgumentParser(description="Unsupervised distillation")

parser.add_argument(
    "--splits", type=str, default="", help="splits of unlabeled data to be labeled"
)

def main():
    a = 1


def get_train_loader(current_indices, data):
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    x = DataLoader(
            CIFAR10Subset(data, transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), current_indices),
            batch_size=128, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    return x

def print_target(indices, start, loop):
    for j in range(start, loop):
        sample = np.load("/root/autodl-tmp/pycode/indices/{}_cifar10_{}.npy".format(indices, j * 100 + 100))
        train_loader = get_train_loader(sample, "/root/autodl-tmp/pycode/data")

        targets = []

        for i, (_, target, _) in enumerate(train_loader):
            targets.append(target)
            print(target)
        targets = torch.tensor(targets, dtype=torch.float)
        out, num = torch.unique(targets, return_counts=True)
        num = torch.tensor(num, dtype=torch.float)
        num = num.reshape(1, 10)
        print(out)
        out = F.normalize(num, dim=1)
        print(out.max() / out.min())


# def test_train(task_model,train_loader):
#     task_model.train()
#     for i, (images, target, _) in enumerate(train_loader):

#         images = images.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)
#         temp = torch.unique(target, return_counts=True)
#         target_list = []
#         index = 0
#         for i in range(10):
#             if temp[0][index] != i:
#                 target_list.append(0)
#             else:
#                 target_list.append(temp[1][index])
#                 index = index + 1

#         samples_per_cls = torch.tensor(target_list)

#         samples_per_cls = samples_per_cls.data.cpu()

#         # compute output
#         output = task_model(images)

#         print(output)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == '__main__': 

    sample = np.load("/root/autodl-tmp/pycode/indices/0.99_kmeans_cifar10_500.npy")

    train_loader = get_train_loader(sample, "/root/autodl-tmp/pycode/data")

    task_model = get_my_backbone_model("resnet18","/root/autodl-tmp/pycode/al/ckpt.pth")
    task_model = torch.nn.DataParallel(task_model).cuda()

    checkpoint = torch.load("/root/autodl-tmp/pycode/0.99_kmeans_cifar10_500.pth.tar")
    task_model.load_state_dict(checkpoint["state_dict"])

    task_model.eval()

    for i, (images, target, _) in enumerate(train_loader):
        # images = images.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        # 计算output
        output = task_model(images)

        temp = torch.unique(target, return_counts=True)
        target_list = []
        index = 0
        for j in range(10):
            if len(temp[0]) > index and temp[0][index] == j:
                target_list.append(temp[1][index])
                index = index + 1
            else:
                target_list.append(0)

        samples_per_cls = torch.tensor(target_list)

        # samples_per_cls = samples_per_cls.data.cpu()

        loss = cb_(target.detach().numpy(), output.cpu().detach().numpy(), samples_per_cls, 10, 0.99)
        
        print(loss)
        loss = CB_loss(target,output,samples_per_cls,10,'softmax',0.99)
        print(loss)
        
        print(images.shape)
        output = task_model(images)
        print(output.shape)
        print(target.shape)
#     sample = np.load("/root/autodl-tmp/pycode/indices/kmeans_cifar10_200.npy")
#     train_loader = get_train_loader(sample, "/root/autodl-tmp/pycode/data")

#     task_model = get_my_backbone_model("resnet18","/root/autodl-tmp/pycode/al/ckpt.pth")

#     task_model = torch.nn.DataParallel(task_model).cuda()


#     optimizer = torch.optim.SGD(
#         task_model.parameters(),
#         0.001,
#         momentum=0.9,
#         weight_decay=1e-4,
#     )

    # writer = SummaryWriter('./runs/log')

    # task_model.train()

    # losses = 0
    # for epoch in range(10):
    #     for i, (images, target, _) in enumerate(train_loader):

    #         images = images.cuda(non_blocking=True)
    #         target = target.cuda(non_blocking=True)

    #         input = task_model(images)

    #         temp = torch.unique(target, return_counts=True)
    #         target_list = []
    #         index = 0
    #         for j in range(10):
    #             if len(temp[0]) > index or temp[0][index] != j:
    #                 target_list.append(0)
    #             else:
    #                 target_list.append(temp[1][index])
    #                 index = index + 1

    #         samples_per_cls = torch.tensor(target_list)

    #         samples_per_cls = samples_per_cls.data.cpu()

    #         effective_num = 1.0 - np.power(0.99, samples_per_cls)
    #         effective_num = np.array(effective_num)
    #         effective_num = np.maximum(effective_num, np.finfo(np.float32).eps)
    #         weights = (1.0 - 0.99) / effective_num
    #         weights = weights / np.sum(weights) * 4

    #         loss = F.cross_entropy(input,target,torch.tensor(weights,dtype=torch.float).cuda(non_blocking=True))
            
    #         acc1, acc5 = accuracy(input, target, topk=(1, 5))
    #         losses = losses + loss.item()

    #         writer.add_custom_scalars("train_loss", loss,epoch)
    #         writer.add_custom_scalars("total_losses",losses,epoch)
    #         writer.add_custom_scalars("acc1",acc1[0],epoch)
    #         writer.add_custom_scalars("acc5",acc5[0],epoch)

    #         # pred = input.softmax(dim = 1)
    #         # cb_loss = F.binary_cross_entropy(input = input, target = target, weight = weights)
    #         # compute gradient and do SGD step
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    # writer.add_graph(task_model,images)
