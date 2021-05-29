import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim
import matplotlib.pyplot as plt
import skimage
import skimage.transform
from skimage.color import rgb2gray
from skimage.io import imread_collection
from copy import deepcopy
from utils import *

ways = 5
shots = 5
meta_lr = 0.03
fast_lr = 0.5
meta_batch_size = 32
adaptation_steps = 1
num_iterations = 6000
cuda = True
seed = 42
meta_test_error = 0.0
meta_test_accuracy = 0.0

tasksets = l2l.vision.benchmarks.get_tasksets('mini-imagenet',
    train_samples=2 * shots,
    train_ways=ways,
    test_samples=2 * shots,
    test_ways=ways,
    root='../data',
)

device = torch.device('cpu')
if cuda and torch.cuda.device_count():
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda')

model = l2l.vision.models.MiniImagenetCNN(ways)
model.load_state_dict(torch.load('../out/5-way5-shot/model5999'))
model.to(device)

maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
loss = nn.CrossEntropyLoss(reduction='mean')


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

print('Meta Test Error', meta_test_error / meta_batch_size)
print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


