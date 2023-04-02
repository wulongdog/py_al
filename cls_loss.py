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

import random
from typing import Optional, List
import torch
import torch.nn.functional as F
import tensorflow as tf

import numpy as np


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    cb_loss = None
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    effective_num = np.array(effective_num)
    effective_num = np.maximum(effective_num, np.finfo(np.float32).eps)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels.long(), no_of_classes).float()
    labels_one_hot = labels_one_hot.cuda(non_blocking=True)

    init_weight = torch.tensor(weights,dtype=torch.float)
    init_weight = init_weight.cuda(non_blocking=True)

    weights = torch.tensor(weights,dtype=torch.float)

    weights = weights.cuda(non_blocking=True)

    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)
    # weights = F.normalize(weights, p=2, dim=1)
  
    if loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        logits = logits.softmax(dim=1)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        cb_loss = loss_fn(logits, labels_one_hot) * weights
        # cb_loss = F.cross_entropy(input=logits, target=labels_one_hot, weight=weights)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=init_weight)
        # loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        # logits = logit_adjustment(logits,[x / sum(samples_per_cls) for x in samples_per_cls],1.0)
        # cb_loss = loss_fn(input=logits, target=labels_one_hot)
        # cb_loss = loss_fn(input=logits, target=labels)
    return cb_loss.mean()

def logit_adjustment(
        outputs: torch.Tensor,
        prior: List[float] = None,
        tau: Optional[float] = 1.0,
):
    log_prior = torch.log(torch.tensor(prior, dtype=torch.float) + 1e-8).to(torch.device('cuda'))
    log_prior = log_prior.expand(outputs.size()[0], -1)
    outputs = outputs + log_prior * tau
    return outputs

def cb_(label, logits, samples_per_cls, num_classes, beta):
    
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    effective_num = np.array(effective_num)
    effective_num = np.maximum(effective_num, np.finfo(np.float32).eps)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * num_classes
    
    one_hot_labels = tf.one_hot(label, num_classes)

    weights = tf.cast(weights, dtype=tf.float32)
    weights = tf.expand_dims(weights, 0)
    weights = tf.tile(weights, [tf.shape(one_hot_labels)[0], 1]) * one_hot_labels
    weights = tf.reduce_sum(weights, axis=1)
    weights = tf.expand_dims(weights, 1)
    weights = tf.tile(weights, [1, num_classes])

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,label_smoothing=0)

    tower_loss = cce(
        one_hot_labels, logits, sample_weight=tf.reduce_mean(weights, axis=1))
    tower_loss = tf.reduce_mean(tower_loss)

    return tower_loss
	