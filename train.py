#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load mini-ImageNet, and
    * sample tasks and split them in adaptation and evaluation sets.

To contrast the use of the benchmark interface with directly instantiating mini-ImageNet datasets and tasks, compare with `protonet_miniimagenet.py`.
"""

import random
import numpy as np
import os

import torch
from torch import nn, optim

import learn2learn as l2l
from learn2learn.data.transforms import (NWays,
                                         KShots,
                                         LoadData,
                                         RemapLabels,
                                         ConsecutiveLabels)
import logging
import datetime
from utils import *


def main(
        ways=5,
        shots=5,
        meta_lr=0.03,
        fast_lr=0.5,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=6000,
        cuda=True,
        seed=42,
):
    logPath = str(ways) + '-ways' + str(shots) + '-shots.' + 'out.log'
    logging.basicConfig(filename=logPath, level=logging.DEBUG)
    logging.info(datetime.datetime.now())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
                                                  train_samples=2 * shots,
                                                  train_ways=ways,
                                                  test_samples=2 * shots,
                                                  test_ways=ways,
                                                  root='data',
                                                  )

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

        # if iteration % 100 == 0 or iteration == num_iterations - 1:
        #     logging.info(datetime.datetime.now())
        #     logging.info('Iteration %s', str(iteration))
        #     logging.info('Meta Train Error %s', str(meta_train_error / meta_batch_size))
        #     logging.info('Meta Train Accuracy %s', str(meta_train_accuracy / meta_batch_size))
        #     logging.info('Meta Valid Error %s', str(meta_valid_error / meta_batch_size))
        #     logging.info('Meta Valid Accuracy %s', str(meta_valid_accuracy / meta_batch_size))
        #     logging.info('\n')
        #
        #     path = "./out/" + str(ways) + "-way" + str(shots) + "-shot"
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     torch.save(model.state_dict(), path + "/model" + str(iteration))

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()
        print('Iteration', iteration)

    logging.info(datetime.datetime.now())

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = maml.clone()
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    logging.info(datetime.datetime.now())
    logging.info('Meta Test Error %s', str(meta_test_error / meta_batch_size))
    logging.info('Meta Test Accuracy %s', str(meta_test_accuracy / meta_batch_size))
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main()
