import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LinearClsHead', 'AngularClsHead']


class LinearCls(nn.Module):
    def __init__(self, 
                 num_classes,
                 in_channels,
                 hidden_dim,
                 softmax=True,
                 sigmoid=False):
        super(LinearCls, self).__init__()

        if type(num_classes) == int:
            num_classes = num_classes
        elif type(num_classes) == tuple and len(num_classes) == 1:
            num_classes = num_classes[0]
        else:
            raise ValueError

        self.softmax = softmax
        self.sigmoid = sigmoid

        self.fc0 = nn.Linear(in_channels, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.init_weights()


    def init_weights(self):
        def normal_init(module, mean=0, std=1, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.normal_(module.weight, mean, std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.Linear)):
                normal_init(m, mean=1, std=0.01, bias=0)


    def forward(self, x):
        x = self.fc0(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc(x)
        if self.softmax:
            cls_score = F.log_softmax(x, dim=-1)
        elif self.sigmoid:
            cls_score = F.sigmoid(x)
        else:
            cls_score = x
        return cls_score



class LinearClsHead(nn.Module):
    def __init__(self, 
                 num_classes,
                 in_channels,
                 ):
        super(LinearClsHead, self).__init__()

        if type(num_classes) == int:
            num_classes = num_classes
        elif type(num_classes) == tuple and len(num_classes) == 1:
            num_classes = num_classes[0]
        else:
            raise ValueError

        self.fc = nn.Linear(in_channels, num_classes)
        self.init_weights()


    def init_weights(self):
        def normal_init(module, mean=0, std=1, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.normal_(module.weight, mean, std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.Linear)):
                normal_init(m, mean=1, std=0.01, bias=0)


    def forward(self, x):
        x = F.relu(x)
        x = self.fc(x)
        cls_score = F.log_softmax(x, dim=-1)
        return cls_score



class AngularClsHead(nn.Module):
    def __init__(self, 
                 num_classes,
                 in_channels,
                 m):
        super(AngularClsHead, self).__init__()

        if type(num_classes) == int:
            num_classes = num_classes
        elif type(num_classes) == tuple and len(num_classes) == 1:
            num_classes = num_classes[0]
        else:
            raise ValueError

        self.fc = nn.Linear(in_channels, num_classes)
        self.m = m
        self.mlambda = [
                lambda x: x**0,
                lambda x: x**1,
                lambda x: 2*x**2-1,
                lambda x: 4*x**3-3*x,
                lambda x: 8*x**4-8*x**2+1,
                lambda x: 16*x**5-20*x**3+5*x
        ]
        self.init_weights()


    def init_weights(self):
        def normal_init(module, mean=0, std=1, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.normal_(module.weight, mean, std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.Linear)):
                normal_init(m, mean=1, std=0.01, bias=0)


    def forward(self, x):

        w = torch.transpose(self.fc.weight, 0, 1) # size=(F,Classnum) F=in_features Classnum=out_features
        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum
        cos_theta = x.mm(ww) # size=(B,Classnum)

        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        # theta = torch.cuda.FloatTensor(cos_theta.data.acos())
        theta = torch.FloatTensor(cos_theta.data.acos())
        k = (self.m*theta/3.14159265).floor()
        n_one = k*0.0 - 1
        phi_theta = (n_one**k) * cos_m_theta - 2*k
        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        
        return (cos_theta, phi_theta)
