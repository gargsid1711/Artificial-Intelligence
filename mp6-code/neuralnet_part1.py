# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
        self.hidden = nn.Linear(self.in_size, 128)
        self.activation = nn.ReLU()
        self.output = nn.Linear(128, 2)
        self.optimizer = torch.optim.SGD(self.parameters(), self.lrate, momentum=0.9)

    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.activation(x)

        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """
    net = NeuralNet(0.0016, nn.CrossEntropyLoss(), len(train_set[0]), len(train_labels))

    means = train_set.mean(dim=1, keepdim=True)
    stds = train_set.std(dim=1, keepdim=True)
    normalized_data = (train_set - means) / stds

    start = 0
    end = batch_size
    final_loss = []
    for epoch in range(n_iter):
        if end >= len(train_set):
            end = len(train_set)
            current_loss = net.step(normalized_data[start:end], train_labels[start:end])
            final_loss.append(current_loss)
            start = 0
            end = batch_size
            continue
        current_loss = net.step(normalized_data[start:end], train_labels[start:end])
        start += batch_size
        end += batch_size
        final_loss.append(current_loss)


    means = dev_set.mean(dim=1, keepdim=True)
    stds = dev_set.std(dim=1, keepdim=True)
    normalized_dev = (dev_set - means) / stds

    output = net(normalized_dev)
    _, predicted = torch.max(output.data, 1)
    predicted = np.array(predicted)

    return final_loss,predicted,net
