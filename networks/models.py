# encoding: utf-8

"""
DenseNet-121
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import densenet

class DenseNet121(nn.Module):
    def __init__(self, out_size, mode, drop_rate=0):
        super(DenseNet121, self).__init__()
        assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet121 = densenet.densenet121(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        
        if mode in ('U-Ones', 'U-Zeros'):
            self.densenet121.classifier = nn.Sequential(
                nn.Linear(num_ftrs, out_size),
            )
        elif mode in ('U-MultiClass', ):
            self.densenet121.classifier = None
            self.densenet121.Linear_0 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_1 = nn.Linear(num_ftrs, out_size)
            self.densenet121.Linear_u = nn.Linear(num_ftrs, out_size)
            
        self.mode = mode
        
        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

        
    def forward(self, x, Pi=None, priors_corr=None):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)
        
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        self.activations = out
        
        
        if self.mode in ('U-Ones', 'U-Zeros'):
            out = self.densenet121.classifier(out)
        elif self.mode in ('U-MultiClass', ):
            n_batch = x.size(0)
            out_0 = self.densenet121.Linear_0(out).view(n_batch, 1, -1)
            out_1 = self.densenet121.Linear_1(out).view(n_batch, 1, -1)
            out_u = self.densenet121.Linear_u(out).view(n_batch, 1, -1)
            out = torch.cat((out_0, out_1, out_u), dim=1)

        g = torch.softmax(out, dim=1)
        if Pi is not None and priors_corr is not None:    
            g = self.TransFunc(g, Pi, priors_corr)
            
        
        return self.activations, out, g


    def TransFunc(self, g, Pi, priors_corr):
      
        pi_ita = torch.mm(Pi, g.permute(1, 0))
        rou_pi_ita = torch.matmul(priors_corr, pi_ita)
        pi_corr = pi_ita.permute(1, 0) * priors_corr.unsqueeze(0)
        output = (pi_corr.permute(1, 0) / rou_pi_ita).permute(1, 0)
        del pi_ita, rou_pi_ita, pi_corr

        return output




class MLP(nn.Module):
    def __init__(self, dim, projection_size = 2048, hidden_size = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)