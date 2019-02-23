import numpy as np
from collections import OrderedDict, namedtuple
from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable


class CNN(nn.Module):
    """
    Container class for convolutional neural network.CNN
    """
    
    def __init__(self, conv_layers, fc_in_shape, final_layers):
        super(CNN, self).__init__()
        
        self.fc_in_shape = fc_in_shape
        
        self.cv_layers = nn.Sequential(conv_layers)
        self.fc_layers = nn.Sequential(final_layers)
        
        self.out_size = final_layers
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        for i, layer in enumerate(self.cv_layers):
            try:
                x = layer(x)
            except RuntimeError:
                print('(CV) Layer %d: %s' % (i, str(layer)))
                print('x.shape: %s' % str(x.shape))
                raise
                
        x = x.view(batch_size, self.fc_in_shape)
        
        for i, layer in enumerate(self.fc_layers):
            try:
                x = layer(x)
            except RuntimeError:
                print('(FC) Layer %d: %s' % (i, str(layer)))
                print('x.shape: %s' % str(x.shape))
                raise
                
        return x
        
    @classmethod
    def from_dict(cls, d):
        raise NotImplementedError
        
    def to_dict(self, d):
        raise NotImplementedError
        
    def predict(self, x):
        n = x.shape[0]
        scores = self(x).detach().numpy()
        
        probs = np.exp(scores).reshape(n, -1)
        probs /= probs.sum(axis=1).reshape(-1, 1)
        
        return probs
    

def get_output_size_2d(layers, in_size):
    """
    Returns the output dimensions of an image after it has been passed through 
    a series of 2D convolutional and pooling layers.
    
    NOTE: ONLY WORKS if all passed layers are either conv/pool layers OR they do not change
            the shape of their inputs.
    """
    H, W = in_size
    
    if isinstance(layers, dict):
        iterator = layers.values()
    else:
        iterator = layers
    
    for layer in iterator:
        if not ('Conv2d' in str(layer) or 'Pool2d' in str(layer)):
            continue
            
        e = lambda i: (i, i) if isinstance(i, int) else i
        
        p0, d0 = e(layer.padding)[0], e(layer.dilation)[0]
        k0, s0 = e(layer.kernel_size)[0], e(layer.stride)[0]
        H = int(np.floor(((H + 2 * p0 - d0 * (k0 - 1) - 1) / s0) + 1))

        p1, d1 = e(layer.padding)[1], e(layer.dilation)[1]
        k1, s1 = e(layer.kernel_size)[1], e(layer.stride)[1]
        W = int(np.floor(((W + 2 * p1 - d1 * (k1 - 1) - 1) / s1) + 1))
        
    return (H, W)


def train_CNN(
    model, train_data, optimizer, loss, num_epochs, batch_size, valid_data=None, verbose=False,
    valid_stop_threshold=0.001
):
    """
    Train convolutional neural network for given number of epochs.
    """
    X_train, y_train = train_data
    n = X_train.shape[0]
    X_train = X_train.reshape(n, 1, 28, 28)
    
    stats = dict()
    stats['train_loss'] = []
    
    if valid_data is not None:
        X_valid, y_valid = valid_data
        X_valid = X_valid.reshape(-1, 1, 28, 28)
        
        stats['valid_acc'] = []
        stats['best_model'] = (deepcopy(model), -np.inf)

    
    num_batches = int(np.ceil(n / batch_size))
    
    try:
        for epoch in range(num_epochs):
            if verbose:
                print('=='*40)
                print('Epoch [%d/%d]' % (epoch, num_epochs))

            shuf_idx = np.random.choice(n, replace=False, size=n)
            
            train_loss = 0.
            valid_acc = 0.
            for batch_num in range(num_batches):
                b = batch_size * batch_num
                e = b + batch_size
                
                X_batch = Variable(torch.Tensor(X_train[shuf_idx[b:e], :]))
                y_true = Variable(torch.LongTensor(y_train[shuf_idx[b:e]]))
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                
                batch_loss = loss(y_pred, y_true)
                batch_loss.backward()
                
                optimizer.step()
                
                train_loss += batch_loss.data
                
            train_loss /= num_batches
            stats['train_loss'].append(train_loss)
            
            if verbose:
                print('Avg train loss: %.5f' % train_loss)
                
            if valid_data is not None:
                model.eval()
                y_pred = model.predict(torch.Tensor(X_valid)).argmax(axis=1)
                model.train()

                num_correct = len(np.argwhere(y_pred.squeeze() == y_valid.squeeze()))
                acc = num_correct / y_valid.shape[0]
                stats['valid_acc'].append(acc)
                
                if acc > stats['best_model'][1]:
                    stats['best_model'] = (deepcopy(model), acc)
                
                if verbose:
                    print('Valid acc: %.3f' % acc)
                
                if epoch > 2 and (np.std(stats['valid_acc'][-5:]) < valid_stop_threshold):
                    if verbose:
                        print('Validation accuracy has plateaued. Returning early.')
                        
                    break
                    
                
    except KeyboardInterrupt:
        if verbose:
            print('Keyboard interrupt. Returning early.')
            
            
    return model, stats





if __name__ == '__main__':
    pass