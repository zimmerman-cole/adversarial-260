"""
Module for white box adversarial attackers.
"""

import numpy as np
import torch
from torch.autograd import Variable

def fgsm(x, y, model, loss_function, projector, eps, num_steps=1):
    assert isinstance(x, np.ndarray), type(x)
    assert isinstance(y, np.ndarray), type(y)
    
    og_x = np.array(x)
    
    for step in range(num_steps):
        x = Variable(torch.Tensor(x.reshape(-1, 1, 28, 28)), requires_grad=True)

        scores = model(x)
        probs = np.exp(scores.detach().numpy()).reshape(-1, 10)
        probs /= probs.sum(axis=1).reshape(-1, 1)

        loss = loss_function(scores, Variable(torch.LongTensor(y)))
        loss.backward()

        grads = torch.sign(x.grad)
        
        old_x = np.array(x.detach().numpy())
        x = x.detach().numpy() + eps * grads.detach().numpy()
        x = projector(x, og_x)  # project back onto constraint space
        
#         print('loss=%.5f' % loss.data)
        
    return x