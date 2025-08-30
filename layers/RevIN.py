import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    
class LeddamRevIN(nn.Module):
    def __init__(self,channel,output_dim):
        super(LeddamRevIN, self).__init__()
        self.output_dim=output_dim
    def forward(self, x):
        # Calculate mean and std along dim=1
        self.means = x.mean(1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        
        # Normalize using learned parameters
        x_normalized = (x - self.means) / self.stdev
        return x_normalized
    
    def inverse_normalize(self, x_normalized):
        x_normalized = x_normalized * \
                        (self.stdev[:, 0, :].unsqueeze(1).repeat(
                            1, self.output_dim, 1))
        x_normalized = x_normalized + \
                            (self.means[:, 0, :].unsqueeze(1).repeat(
                                1, self.output_dim, 1)) 
        return x_normalized