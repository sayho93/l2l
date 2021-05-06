#!/usr/bin/env python3

"""
File: metacurvature_fc100.py
Author: Seb Arnold - seba1511.net
Email: smr.arnold@gmail.com
Github: seba-1511
Description:
Demonstrates how to use the GBML wrapper to implement MetaCurvature.

A demonstration of the low-level API is available in:
    examples/vision/anilkfo_cifarfs.py
"""
import os
import random
import numpy as np
import torch
import learn2learn as l2l
from learn2learn.optim.transforms import MetaCurvatureTransform
import logging
import datetime
from utils import *
from CifarCNN import *


def main(
    fast_lr=0.1,
    meta_lr=0.01,
    num_iterations=10000,
    meta_batch_size=16,
    adaptation_steps=5,
    shots=5,
    ways=5,
    cuda=1,
    seed=1234
    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Tasksets using the benchmark interface
    tasksets = l2l.vision.benchmarks.get_tasksets(
        name='fc100',
        train_samples=2*shots,
        train_ways=ways,
        test_samples=2*shots,
        test_ways=ways,
        root='data',
    )

    logPath = str(ways) + '-ways' + str(shots) + '-shots.FC100.' + 'out.log'
    logging.basicConfig(filename=logPath, level=logging.DEBUG)
    logging.info(datetime.datetime.now())

    # Create model
    model = CifarCNN(output_size=ways)
    model.to(device)
    gbml = l2l.algorithms.GBML(
        model,
        transform=MetaCurvatureTransform,
        lr=fast_lr,
        adapt_transform=False,
    )
    gbml.to(device)
    opt = torch.optim.Adam(gbml.parameters(), meta_lr)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = gbml.clone()
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
            learner = gbml.clone()
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

        if iteration % 100 == 0 or iteration == num_iterations - 1:
            logging.info(datetime.datetime.now())
            logging.info('Iteration %s', str(iteration))
            logging.info('Meta Train Error %s', str(meta_train_error / meta_batch_size))
            logging.info('Meta Train Accuracy %s', str(meta_train_accuracy / meta_batch_size))
            logging.info('Meta Valid Error %s', str(meta_valid_error / meta_batch_size))
            logging.info('Meta Valid Accuracy %s', str(meta_valid_accuracy / meta_batch_size))
            logging.info('\n')

            path = "./out/FC100/" + str(ways) + "-way" + str(shots) + "-shot"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), path + "/model" + str(iteration))

        # Average the accumulated gradients and optimize
        for p in gbml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        learner = gbml.clone()
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
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


if __name__ == '__main__':
    main()
