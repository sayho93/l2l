import os
import random
import numpy as np
import torch
import learn2learn as l2l
import logging
import datetime
from CifarCNN import *
from utils import *

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
        train_samples=2 * shots,
        train_ways=ways,
        test_samples=2 * shots,
        test_ways=ways,
        root='data',
    )

    logPath = str(ways) + '-ways' + str(shots) + '-shots.CIFARFS.' + 'out.log'
    logging.basicConfig(filename=logPath, level=logging.INFO)
    logging.info(datetime.datetime.now())

    model = CifarCNN(output_size=ways)
    model.load_state_dict(torch.load("out/CIFARFS/5-way5-shot/model9999"))
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

    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        # Compute meta-testing loss
        task_features = l2l.clone_module(features)
        task_classifier = l2l.clone_module(classifier)
        task_update = l2l.clone_module(fast_update)
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt_cifarfs(batch,
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
    logging.info('Meta Test Error %s', str(meta_test_error / meta_batch_size))
    logging.info('Meta Test Accuracy %s', str(meta_test_accuracy / meta_batch_size))


if __name__ == '__main__':
    main()



