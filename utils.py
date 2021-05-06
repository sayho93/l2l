import torch
import numpy as np

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy

def fast_adapt_cifarfs(
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