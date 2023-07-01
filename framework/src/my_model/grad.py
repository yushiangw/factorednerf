

import torch
import torch.nn as nn
import torch.nn.functional as F 


def gradient(output, input, grad_outputs=None, graph=True):
     
    if grad_outputs is None:
        grad_outputs = torch.ones_like(output)

    grad = torch.autograd.grad(output, [input], 
                               grad_outputs=grad_outputs,
                               retain_graph=True,
                               create_graph=graph)[0]
    return grad

def gradient_list(output, input_list, grad_outputs=None, graph=True):
     
    if grad_outputs is None:
        grad_outputs = torch.ones_like(output)

    grad = torch.autograd.grad(output, input_list, 
                               grad_outputs=grad_outputs,
                               retain_graph=True,
                               create_graph=graph)
    return grad


def num_grad():
    # numerical_gradient
    d_steps=torch.tensor([ [ 1.,  0.,  0.],
                           [-1.,  0.,  0.],
                           [ 0.,  1.,  0.],
                           [ 0., -1.,  0.],
                           [ 0.,  0.,  1.],
                           [ 0.,  0., -1.]])

    d_steps=d_steps.reshape(-1,3)

    return d_steps

def num_grad2d():
    # numerical_gradient
    d_steps=torch.tensor([ [ 1.,  0.],
                           [-1.,  0.],
                           [ 0.,  1.],
                           [ 0., -1.]])

    d_steps=d_steps.reshape(-1,2)

    return d_steps


def cube_nn_step():

    d_steps=torch.tensor([
                           [ 1.,   1.,   1.],
                           [ 1. ,  1.,  -1.],
                           [ 1.,  -1.,   1.],
                           [ -1.,  1.,   1.],
                           [ 1.,  -1.,  -1.],
                           [ -1., -1.,   1.],
                           [ -1.,  1.,  -1.],
                           [ -1., -1.,  -1.],
                         ])

    d_steps=d_steps.reshape(-1,3)

    return d_steps

def set_module_grad(model, val):
  for param in model.parameters():
      param.requires_grad = val