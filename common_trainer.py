# encoding: utf-8
import argparse
import os
import random
from re import split
import smtplib
import time

import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from cls_loss import CB_loss, cb_, logit_adjustment
from plt import plt_save

import strategies
from custom_datasets import *
import sampler
import trainer_DP
import train_model
import smtplib
from email.mime.text import MIMEText

parser = argparse.ArgumentParser(description="普通主动学习相关参数")
parser.add_argument("data", metavar="DIR", help="数据集的路径")
parser.add_argument(
    "--dataset", default="cifar10", type=str, help="数据集名称"
)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="数据加载类dataloader workers (default: 4)",
)
parser.add_argument("-a", "--arch", default="resnet18", help="使用的预训练模型")
parser.add_argument(
    "--epochs", default=400
    , type=int, metavar="N", help="训练的总轮数"
)
parser.add_argument(
    "--stop-count", default=5
    , type=int, metavar="N", help="early stopping counter"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="开始训练的轮数",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="dataloader -batch size (default: 128), "
    "对于每次取样的数据集使用 Data Parallel 时的batch size"
,
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="初始化学习率",
    dest="lr",
)
parser.add_argument(
    "--lr_schedule", type=str, default="30,60,90", help="lr drop schedule"
)
parser.add_argument(
    "--weights",
    dest="weights",
    type=str,
    required=True,
    help="预训练模型 weights",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="优化器 momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="优化器 weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="打印频率 (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="要恢复的checkpoint的路径 (default: none)",
)
parser.add_argument(
    "--resume-indices",
    default="",
    type=str,
    metavar="PATH",
    help="要恢复的数据indices (default: none)",
)
parser.add_argument(
    "--seed", default=1234, type=int, help="随机种子seed 用来初始化训练 "
)
parser.add_argument(
    "--save", default="output", type=str, help="inference_feats输出目录"
)
parser.add_argument(
    "--indices", default="indices", type=str, help="取样的样本目录"
)
parser.add_argument(
    "--splits", type=int, default=100, help="每次去unlabel data的数量"
)
parser.add_argument(
    "--name", type=str, default="kmeans", help="取样的方法名称"
)
parser.add_argument('--load-cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('--start-budget', type=int, default=100, help="现在训练的样本数量")
parser.add_argument('--budget', type=int, default=500, help="训练的总样本数量")
parser.add_argument(
    "--backbone",
    type=str,
    default="compress",
    help="要进行线性计算的方法的名称",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.99,
    help="beta",
)
parser.add_argument(
    "--common",
    type=bool,
    default=False,
    help="beta",
)
parser.add_argument(
    '-bd','--base-dir', default="", type=str, help="基本工作目录"
)
parser.add_argument(
    '--tensorboard-dir', default="runs/log", type=str, help="tensorboard日志目录"
)
parser.add_argument(
    '--plt-dir', default="plt", type=str, help="plt日志目录"
)

def send_mail(message):
    #设置服务器所需信息
    #163邮箱服务器地址
    mail_host = 'smtp.163.com'  
    #163用户名
    mail_user = 'longdogwu@163.com'  
    #密码(部分邮箱为授权码) 
    mail_pass = 'FTTHKJAVKFWUYEIN'   
    #邮件发送方邮箱地址
    sender = 'longdogwu@163.com'  
    #邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
    receivers = ['2218094687@qq.com']  

    #设置email信息
    #邮件内容设置
    message = MIMEText(message,'plain','utf-8')
    #邮件主题       
    message['Subject'] = '训练完成了' 
    #发送方信息
    message['From'] = sender 
    #接受方信息     
    message['To'] = receivers[0]  

    #登录并发送邮件
    try:
        smtpObj = smtplib.SMTP() 
        #连接到服务器
        smtpObj.connect(mail_host,25)
        #登录到服务器
        smtpObj.login(mail_user,mail_pass) 
        #发送
        smtpObj.sendmail(
            sender,receivers,message.as_string()) 
        #退出
        smtpObj.quit() 
        print('success')
    except smtplib.SMTPException as e:
        print('error',e) #打印错误

def main():
    args = parser.parse_args()

    # args.batch_size = int(args.splits / 10)

    if args.base_dir == "":
        args.base_dir = os.path.dirname(__file__)

    if not os.path.exists("{}/{}".format(args.base_dir,args.indices)):
        os.makedirs("{}/{}".format(args.base_dir,args.indices))
    if not os.path.exists("{}/{}".format(args.base_dir,args.save)):
        os.makedirs("{}/{}".format(args.base_dir,args.save))
    if not os.path.exists("{}/{}".format(args.base_dir,args.plt_dir)):
        os.makedirs("{}/{}".format(args.base_dir,args.plt_dir))

    writer = SummaryWriter("{}/{}".format(args.base_dir,args.tensorboard_dir))

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

    elif args.dataset == "fashionmnist":
        args.num_images = 60000
        args.num_classes = 10


    else:
        raise NotImplementedError

    # 设置初始化模型参数
    init_args = copy.deepcopy(args)
    init_args.name = "random"
    init_args.resume = ""

    # 设置随机种子
    seed = int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    init_tensorboard(args,writer)
    inference_feats, inference_labels = get_inference(args)
    val_loader = trainer_DP.get_val_loader(args.dataset, args)

    # 加载模型
    print("=> creating model")
    task_model = train_model.get_backbone_model(args)
    task_model = torch.nn.DataParallel(task_model).cuda()

    init_module(init_args, val_loader,inference_feats, inference_labels,task_model,writer)
    # 训练
    main_work(args, val_loader, inference_feats, inference_labels,task_model,writer)
    # 训练完成邮件通知我
    send_mail("{}-{}-{}-{}-{} 训练完成".format(
            args.beta,args.dataset,args.name,args.budget,time.asctime(time.localtime(time.time()))
            ))

def init_tensorboard(args,writer): 
    backbone = train_model.get_backbone_model(args)
    if args.dataset == "imagenet":
        args.num_images = 1281167
        args.num_classes = 1000

    elif args.dataset == "imagenet_lt":
        args.num_images = 115846
        args.num_classes = 1000

    elif args.dataset == "cifar100":
        args.num_images = 50000
        args.num_classes = 100
        writer.add_graph(backbone,torch.rand(128,3,32,32))

    elif args.dataset == "cifar10":
        args.num_images = 50000
        args.num_classes = 10
        writer.add_graph(backbone,torch.rand(128,3,32,32))

    elif args.dataset == "fashionmnist":
        args.nu_images = 60000
        args.num_classes = 10
        writer.add_graph(backbone,torch.rand(128,3,28,28))

# 随机取样初始训练
def init_module(init_args, val_loader, inference_feats, inference_labels,task_model,writer):
    print("init module start......")
    # 先random选出初始训练集
    sampler_fc(init_args, inference_feats, inference_labels,task_model)
    # 初始训练
    train_fc(init_args, val_loader,task_model,writer)


def get_inference(args):
    all_indices = np.arange(args.num_images)

    inference_loader = sampler.get_inference_loader(args.dataset, all_indices, args)
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


def sampler_fc(args, inference_feats, inference_labels,task_model):
    all_indices = np.arange(args.num_images)

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

    # 加载之前的数据
    current_indices = []
    save_indices = []
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
    elif os.path.isfile(
        "{}/{}/{}_random_{}_{}.npy".format(args.base_dir,args.indices,args.beta,args.dataset, args.splits)
    ):
        print(
            "=> Loading current indices: {}/{}/{}_random_{}_{}.npy".format(
                args.base_dir,args.indices,args.beta,args.dataset, args.splits
            )
        )
        current_indices = np.load(
            "{}/{}/{}_random_{}_{}.npy".format(args.base_dir,args.indices,args.beta,args.dataset, args.splits)
        )
        save_indices = np.load(
            "{}/{}/{}_random_{}_{}.npy".format(args.base_dir,args.indices,args.beta,args.dataset, args.splits)
        )
        print(
            "current indices size: {}. {}% of all categories is seen".format(
                len(current_indices),
                len(np.unique(inference_labels[current_indices]))
                / args.num_classes
                * 100,
            )
        )

    unlabeled_indices = np.setdiff1d(all_indices, save_indices)
    print("Current unlabeled indices is {}.".format(len(unlabeled_indices)))

    if args.name == "uniform":
        print("Query sampling with {} started ...".format(args.name))
        strategies.uniform(inference_labels, args.splits, args)
        return

    elif args.name == "random":
        print("Query sampling with {} started ...".format(args.name))
        save_indices = []
        current_indices = strategies.myrandom(all_indices, inference_labels, args.splits, args)


    elif args.name == "kmeans":
        print("Query sampling with {} started ...".format(args.name))
        current_indices = strategies.fast_kmeans(
            inference_feats, args.splits, args
        )

    elif args.name == "accu_kmeans":
        print("Query sampling with {} started ...".format(args.name))
        sampled_indices = strategies.accu_kmeans(
            inference_feats, args.splits, unlabeled_indices, args
        )
        current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)

    elif args.name == "coreset":
        print("Query sampling with {} started ...".format(args.name))
        sampled_indices = strategies.core_set(
            inference_feats[unlabeled_indices],
            inference_feats[current_indices],
            unlabeled_indices,
            args.splits,
            args,
        )
        current_indices = np.concatenate((current_indices, sampled_indices), axis=-1)
    elif args.name == "entropy":
        print("Query sampling with {} started ...".format(args.name))
        # unlabeled_loader = trainer_DP.get_train_loader(args.dataset,unlabeled_indices,args)
        unlabeled_loader = sampler.get_inference_loader(args.dataset,unlabeled_indices,args)
        sampled_indices = strategies.my_entropy(unlabeled_loader, unlabeled_indices, task_model, args.splits)
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
    save_indices = save_indices.astype(np.int32)
    print(
        "{}/{}/{}_{}_{}_{}.npy".format(
            args.base_dir,args.indices,args.beta, args.name, args.dataset, len(save_indices)
        ) 
    )
    np.save(
        "{}/{}/{}_{}_{}_{}.npy".format(
            args.base_dir,args.indices,args.beta, args.name, args.dataset, len(save_indices)
        ),save_indices
    )

def get_channels(arch):
    if arch == 'alexnet':
        c = 4096
    elif arch == 'pt_alexnet':
        c = 4096
    elif arch == 'resnet50':
        c = 2048
    elif arch == 'resnet18':
        c = 512
    elif arch == 'mobilenet':
        c = 1280
    elif arch == 'resnet50x5_swav':
        c = 10240
    else:
        raise ValueError('arch not found: ' + arch)
    return 

def train_fc(args, val_loader, task_model,writer):

    optimizer = torch.optim.SGD(
        task_model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    sched = [int(x) for x in args.lr_schedule.split(',')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sched
    )

    # 加载之前的模型参数
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))

    #         checkpoint = torch.load(args.resume)

    #         args.start_epoch = checkpoint["epoch"]
    #         best_acc1 = checkpoint["best_acc1"]
    #         acc1 = checkpoint["best_acc1"]
    #         task_model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         print(
    #             "=> loaded checkpoint '{}' (epoch {})".format(
    #                 args.resume, checkpoint["epoch"]
    #             )
    #         )
    #     elif os.path.isfile(
    #         "{}/{}_random_{}_{}.pth.tar".format(args.base_dir,args.beta,args.dataset, args.splits)
    #     ):
    #         print(
    #             "=> loading checkpoint '{}/{}_random_{}_{}.pth.tar'".format(
    #             args.base_dir,args.beta,args.dataset, args.splits)
    #         )

    #         checkpoint = torch.load(
    #             "{}/{}_random_{}_{}.pth.tar".format(args.base_dir,args.beta,args.dataset, args.splits)
    #         )

    #         args.start_epoch = checkpoint["epoch"]
    #         best_acc1 = checkpoint["best_acc1"]
    #         acc1 = checkpoint["best_acc1"]
    #         task_model.load_state_dict(checkpoint["state_dict"])
    #         optimizer.load_state_dict(checkpoint["optimizer"])
    #         print(
    #             "=> loaded checkpoint '{}' (epoch {})".format(
    #                 args.resume, checkpoint["epoch"]
    #             )
    #         )
    # elif os.path.isfile(
    #     "{}/{}_random_{}_{}.pth.tar".format(args.base_dir,args.beta,args.dataset, args.splits)
    # ):
    #     print(
    #         "=> loading checkpoint '{}/{}_random_{}_{}.pth.tar'".format(
    #         args.base_dir,args.beta,args.dataset, args.splits)
    #     )

    #     checkpoint = torch.load(
    #         "{}/{}_random_{}_{}.pth.tar".format(args.base_dir,args.beta,args.dataset, args.splits)
    #     )

    #     args.start_epoch = checkpoint["epoch"]
    #     best_acc1 = checkpoint["best_acc1"]
    #     acc1 = checkpoint["best_acc1"]
    #     task_model.load_state_dict(checkpoint["state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     print(
    #         "=> loaded checkpoint '{}/{}_random_{}_{}.pth.tar' (epoch {})".format(
    #             args.base_dir,args.beta,args.dataset, args.splits, checkpoint["epoch"]
    #         )
    #     )
    # else:
    #     print("=> no checkpoint found at '{}'".format(args.resume))

    best_acc1 = 0

    cudnn.benchmark = True

    current_indices = []
    # 加载数据
    curr_idxs_file = "{}/{}/{}_{}_{}_{}.npy".format(
        args.base_dir,args.indices,args.beta, args.name, args.dataset, args.start_budget
    )
    if os.path.isfile(curr_idxs_file):
        print("=> Loading current indices: {}".format(curr_idxs_file))
        current_indices = np.load(curr_idxs_file)
        print("current indices size: {}.".format(len(current_indices)))
    elif os.path.isfile("{}/{}/{}_random_{}_{}.npy".format(args.base_dir,args.indices, args.beta,args.dataset, args.start_budget)):
        print(
            "=> Loading current indices: '{}/{}/{}_random_{}_{}.npy'".format(
            args.base_dir,args.indices,args.beta, args.dataset, args.start_budget
            )
        )
        current_indices = np.load(
            "{}/{}/{}_random_{}_{}.npy".format(args.base_dir,args.indices, args.beta,args.dataset, args.start_budget)
        )
        print("current indices size: {}.".format(len(current_indices)))
    else:
        print("=> no such file found at '{}'".format(curr_idxs_file))
        print(
            "=> no such file found at '{}/{}/{}_random_{}_{}.npy'".format(
            args.base_dir,args.indices,args.beta, args.dataset, args.start_budget
            )
        )

    # 获得数据集
    train_loader = trainer_DP.get_train_loader(args.dataset, current_indices, args)

    # 训练中途的数据保存,用来画图
    plt_train_save = plt_save()
    plt_test_save = plt_save()

    print("Training task model started ...")
    args.start_epoch = 0
    print("start epoch is{} and epoch is {}".format(args.start_epoch, args.epochs))
    es = 0
    for epoch in range(args.start_epoch, args.epochs):
        # 训练 epochs 次
        train(train_loader, task_model, optimizer, epoch, args,writer,plt_train_save)

        # 验证模型
        
    acc1 = validate(task_model, val_loader)
    plt_train_save.save(args,"train")
    plt_test_save.save(args,"test")

    print("Final accuracy of {} labeled data is: {:.2f}".format(args.start_budget, acc1))


def train(train_loader, task_model, optimizer, epoch, args,writer,plt_save):
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
        
        # 计算数据加载时间
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 计算output
        output = task_model(images)
        loss = None
        # 如果是普通训练,用cross_entrop , 不然用 cross_entrop * weight
        if args.common:
            # loss_fn = torch.nn.CrossEntropyLoss()
            # loss = loss_fn(output, target)
            loss = F.cross_entropy(output,target)
        else:   
            temp = torch.unique(target.cpu(), return_counts=True)
            samples_per_cls = np.zeros(args.num_classes)
            samples_per_cls[temp[0]] = samples_per_cls[temp[0]] + temp[1].numpy()

            loss = CB_loss(target,output,samples_per_cls,len(samples_per_cls),'softmax',args.beta)


        # 计算 accuracy 和 记录 loss,losses,top1,top5
        acc1, acc5 = trainer_DP.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        plt_save.update(loss.item(),loss.sum().item(),top1.avg.item(),top5.avg.item())

        writer.add_scalars("{}_{}_{}_{}_{}_train_loss".format(
                args.beta,args.dataset,args.name,args.splits,args.start_budget), 
                {'loss':loss},epoch)
        writer.add_scalars("{}_{}_{}_{}_{}_total_train_losse".format(
                args.beta,args.dataset,args.name,args.splits,args.start_budget),
                {'losses':torch.tensor([losses.sum])},epoch)
        writer.add_scalars("{}_{}_{}_{}_{}_train_acc1".format(
                args.beta,args.dataset,args.name,args.splits,args.start_budget),
                {'acc1':torch.tensor([top1.avg])},epoch)
        writer.add_scalars("{}_{}_{}_{}_{}_train_acc".format(
                args.beta,args.dataset,args.name,args.splits,args.start_budget),
                {'acc5':torch.tensor([top5.avg])},epoch)

        # 计算导数和SGDstep
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算运行时间
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(task_model, val_loader):
    batch_time = trainer_DP.AverageMeter('Time', ':6.3f')
    losses = trainer_DP.AverageMeter('Loss', ':.4e')
    top1 = trainer_DP.AverageMeter('Acc@1', ':6.2f')
    top5 = trainer_DP.AverageMeter('Acc@5', ':6.2f')
    progress = trainer_DP.ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # 转换为eval模型
    task_model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
             # 计算output
            output = task_model(images)

            acc1, acc5 = trainer_DP.accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
             # 计算导数和SGDstep
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def main_work(args, val_loader, inference_feats, inference_labels,task_model,writer):
    print("start main train......")
    while args.budget > args.start_budget:
        print("{} train action......".format(args.start_budget/args.splits))
        args.start_budget = args.start_budget + args.splits
        # 挑选样本
        args.resume_indices = "{}/{}/{}_{}_{}_{}.npy".format(
            args.base_dir,args.indices,args.beta, args.name, args.dataset, args.start_budget-args.splits
        )
        sampler_fc(args, inference_feats, inference_labels,task_model)

        # args.resume = "{}/{}_{}_{}_{}.pth.tar".format(
        #     args.base_dir,args.beta,args.name, args.dataset, args.start_budget-args.splits
        # )
        # 训练模型
        train_fc(args, val_loader,task_model,writer)
        


if __name__ == "__main__":
    main()
