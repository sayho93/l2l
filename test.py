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

    model = l2l.vision.models.MiniImagenetCNN(ways)
    model.to(device)
    model.load_state_dict(torch.load("out/5-way5-shot/model5999"))
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

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