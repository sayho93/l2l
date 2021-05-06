#!/usr/bin/env python3

"""
File: anilkfo_cifarfs.py
Author: Seb Arnold - seba1511.net
Email: smr.arnold@gmail.com
Github: seba-1511
Description:
Demonstrates how to use the low-level differentiable optimization utilities
to implement ANIL+KFC on CIFAR-FS.

A demonstration of the high-level API is available in:
    examples/vision/metacurvature_fc100.py
"""
import os
import random
import numpy as np
import torch
import learn2learn as l2l
import logging
import datetime
from CifarCNN import *
from utils import accuracy


def fast_adapt(
        batch,
        features,
        classifier,
        update,
        diff_sgd,
        loss,
        adaptation_steps,
        shots,
        ways,
        device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    data = features(data)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model & learned update
    for step in range(adaptation_steps):
        adaptation_error = loss(classifier(adaptation_data), adaptation_labels)
        if step > 0:  # Update the learnable update function
            update_grad = torch.autograd.grad(adaptation_error,
                                              update.parameters(),
                                              create_graph=True,
                                              retain_graph=True)
            diff_sgd(update, update_grad)
        classifier_updates = update(adaptation_error,
                                    classifier.parameters(),
                                    create_graph=True,
                                    retain_graph=True)
        diff_sgd(classifier, classifier_updates)

    # Evaluate the adapted model
    predictions = classifier(evaluation_data)
    eval_error = loss(predictions, evaluation_labels)
    eval_accuracy = accuracy(predictions, evaluation_labels)
    return eval_error, eval_accuracy


def main(
    fast_lr=0.1,
    meta_lr=0.003,
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
        name='cifarfs',
        train_samples=2*shots,
        train_ways=ways,
        test_samples=2*shots,
        test_ways=ways,
        root='/data',
    )

    logPath = str(ways) + '-ways' + str(shots) + '-shots.CIFARFS.' + 'out.log'
    logging.basicConfig(filename=logPath, level=logging.DEBUG)
    logging.info(datetime.datetime.now())

    # Create model and learnable update
    model = CifarCNN(output_size=ways)
    model.to(device)
    features = model.features
    classifier = model.linear
    kfo_transform = l2l.optim.transforms.KroneckerTransform(l2l.nn.KroneckerLinear)
    fast_update = l2l.optim.ParameterUpdate(
        parameters=classifier.parameters(),
        transform=kfo_transform,
    )
    fast_update.to(device)
    diff_sgd = l2l.optim.DifferentiableSGD(lr=fast_lr)

    all_parameters = list(model.parameters()) + list(fast_update.parameters())
    opt = torch.optim.Adam(all_parameters, meta_lr)
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            task_features = l2l.clone_module(features)
            task_classifier = l2l.clone_module(classifier)
            task_update = l2l.clone_module(fast_update)
            batch = tasksets.train.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               task_features,
                                                               task_classifier,
                                                               task_update,
                                                               diff_sgd,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            task_features = l2l.clone_module(features)
            task_classifier = l2l.clone_module(classifier)
            task_update = l2l.clone_module(fast_update)
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               task_features,
                                                               task_classifier,
                                                               task_update,
                                                               diff_sgd,
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

            path = "./out/CIFARFS/" + str(ways) + "-way" + str(shots) + "-shot"
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), path + "/model" + str(iteration))

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        for p in fast_update.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        task_features = l2l.clone_module(features)
        task_classifier = l2l.clone_module(classifier)
        task_update = l2l.clone_module(fast_update)
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           task_features,
                                                           task_classifier,
                                                           task_update,
                                                           diff_sgd,
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