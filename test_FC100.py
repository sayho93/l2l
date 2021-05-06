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
        train_samples=2 * shots,
        train_ways=ways,
        test_samples=2 * shots,
        test_ways=ways,
        root='data',
    )

    logPath = str(ways) + '-ways' + str(shots) + '-shots.FC100.' + 'out.log'
    logging.basicConfig(filename=logPath, level=logging.DEBUG)
    logging.info(datetime.datetime.now())

    model = CifarCNN(output_size=ways)
    model.load_state_dict(torch.load("out/FC100/5-way5-shot/model9999"))
    model.to(device)
    gbml = l2l.algorithms.GBML(
        model,
        transform=MetaCurvatureTransform,
        lr=fast_lr,
        adapt_transform=False,
    )
    gbml.to(device)

    loss = torch.nn.CrossEntropyLoss(reduction='mean')

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
    logging.info('Meta Test Error %s', str(meta_test_error / meta_batch_size))
    logging.info('Meta Test Accuracy %s', str(meta_test_accuracy / meta_batch_size))


if __name__ == '__main__':
    main()
