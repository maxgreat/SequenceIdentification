
# coding: utf-8

# In[196]:

from __future__ import print_function

#torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#torchvision
import torchvision.models as models

#image
from PIL import Image
import cv2

#jupyter
from ipywidgets import FloatProgress
from IPython.display import display


#os
import os
import os.path as path
import glob

#math
import math
import numpy as np
import random


# ## ConvLSTM
# 
# #### LSTMCell

# In[197]:

t = Variable(torch.rand(1,256,6,6))
ht = Variable( torch.zeros(1,128,6,6))
ct = Variable( torch.zeros(1,128,6,6))


# In[198]:

import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=False):
        super(ConvLSTMCell, self).__init__()
        
        self.k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride
        
        self.w_i = nn.Parameter(torch.Tensor(4*out_channels, in_channels, kernel_size, kernel_size))
        self.w_h = nn.Parameter(torch.Tensor(4*out_channels, out_channels, kernel_size, kernel_size))
        self.w_c = nn.Parameter(torch.Tensor(3*out_channels, out_channels, kernel_size, kernel_size))

        self.bias = bias
        if bias:
          self.bias_i = Parameter(torch.Tensor(4 * out_channels))
          self.bias_h = Parameter(torch.Tensor(4 * out_channels))
          self.bias_c = Parameter(torch.Tensor(3 * out_channels))
        else:
          self.register_parameter('bias_i', None)
          self.register_parameter('bias_h', None)
          self.register_parameter('bias_c', None)
        
        self.register_buffer('wc_blank', torch.zeros(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        n = 4 * self.in_channels * self.k * self.k
        stdv = 1. / math.sqrt(n)
        
        self.w_i.data.uniform_(-stdv, stdv)
        self.w_h.data.uniform_(-stdv, stdv)
        self.w_c.data.uniform_(-stdv, stdv)
        
        if self.bias:
            self.bias_i.data.uniform_(-stdv, stdv)
            self.bias_h.data.uniform_(-stdv, stdv)
            self.bias_c.data.uniform_(-stdv, stdv)

        
    def forward(self, x, hx):
        h, c = hx
        wx = F.conv2d(x, self.w_i, self.bias_i, padding=self.padding, stride=self.stride)
        wh = F.conv2d(h, self.w_h, self.bias_h, padding=self.padding, stride=self.stride)
        wc = F.conv2d(c, self.w_c, self.bias_c, padding=self.padding, stride=self.stride)
        
        
        #wc = torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)
        
        i = F.sigmoid(wx[:, :self.out_channels] + wh[:, :self.out_channels] + wc[:, :self.out_channels])
        f = F.sigmoid(wx[:, self.out_channels:2*self.out_channels] + wh[:, self.out_channels:2*self.out_channels] 
                + wc[:, self.out_channels:2*self.out_channels])
        g = F.tanh(wx[:, 2*self.out_channels:3*self.out_channels] + wh[:, 2*self.out_channels:3*self.out_channels])
        """
        
        wxhc = wx + wh + torch.cat((wc[:, :2 * self.out_channels], Variable(self.wc_blank).expand(wc.size(0), wc.size(1) // 3, wc.size(2), wc.size(3)), wc[:, 2 * self.out_channels:]), 1)
    
        i = F.sigmoid(wxhc[:, :self.out_channels])
        f = F.sigmoid(wxhc[:, self.out_channels:2 * self.out_channels])
        g = F.tanh(wxhc[:, 2 * self.out_channels:3 * self.out_channels])
        o = F.sigmoid(wxhc[:, 3 * self.out_channels:])
        """

        c_t = f * c + i * g
        o_t = F.sigmoid(wx[:, 3*self.out_channels:] + wh[:, 3*self.out_channels:] 
                        + wc[:, 2*self.out_channels: ]*c_t)
        h_t = o_t * F.tanh(c_t)
        
        return h_t, (h_t, c_t)


# In[199]:

"""
    Test convLSTM Cell
"""
def testconvLSTMCell():
    c = ConvLSTMCell(256,128,3)
    o = c(t, (ht,ct))
    print(o[0].size() == torch.Size([1,128,6,6]))
    return o


# In[200]:

from torch.nn import init

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


# In[201]:

"""
    Test GruCell
"""
def testGruCell():
    c = ConvGRUCell(256,128,3)
    o = c(t, Variable( torch.zeros(1,128,6,6)))
    print(o.size() == torch.Size([1,128,6,6]))
    return o


# #### ConvRNN

# In[218]:

class convRNN_1_layer(nn.Module):
    """
        Define a RNN with 1 recurrent layer
        args : r_type : lstm | gru
    """
    def __init__(self, r_type="lstm", copyParameters=False):
        super(convRNN_1_layer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.isCuda = False
        self.r_type = r_type
        if r_type == "lstm":
            self.convRNN = ConvLSTMCell(256,128,kernel_size=3, padding=1, stride=1)
        elif r_type == "gru":
            self.convRNN = ConvGRUCell(256,128,3)
        else:
            print("Error : r_type")
            return -1
        
        self.classifier = nn.Sequential(
            nn.Conv2d(128,6,kernel_size=1, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        )
        
        if copyParameters:
            self.copyParameters(models.alexnet(pretrained=True))
        
        
    def copyParameters(self, alexnetModel):
        for i, a in enumerate(alexnetModel.features):
            if type(a) is torch.nn.modules.conv.Conv2d:
                self.features[i].weight = a.weight
                self.features[i].bias   = a.bias
    
    def cuda(self, device=None):
        self.isCuda = True
        return self._apply(lambda t: t.cuda(device))
    
    def forward(self, x):
        batchSize = len(x[0])
        if self.r_type=="lstm":
            outputs = []
            
            if self.isCuda:
                ht = Variable( torch.zeros(batchSize,128,6,6)).cuda()
                ct = Variable( torch.zeros(batchSize,128,6,6)).cuda()
            else:
                ht = Variable( torch.zeros(batchSize,128,6,6))
                ct = Variable( torch.zeros(batchSize,128,6,6))
                
            for i in x:
                xt = self.features(i)
                o, (ht,ct) = self.convRNN(xt, (ht, ct))
                outputs.append(o)
        
        elif self.r_type=="gru":
            outputs = []
            if self.isCuda:
                ht = Variable( torch.zeros(batchSize,128,6,6)).cuda()
            else:
                ht = Variable( torch.zeros(batchSize,128,6,6))
            
            for e,i in enumerate(x):
                xt = self.features(i)
                ht = self.convRNN(xt, ht)
                outputs.append(ht)
        
        return self.classifier(outputs[-1]).view(batchSize,-1)
        #x = self.classifier(x).squeeze().unsqueeze(0)
        


# In[237]:

class convRNN_2_layer(nn.Module):
    """
        Define a RNN with 1 recurrent layer
        args : r_type : lstm | gru
    """
    def __init__(self, r_type="lstm", copyParameters=False):
        super(convRNN_2_layer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.isCuda = False
        self.r_type = r_type
        if r_type == "lstm":
            self.convRNN1 = ConvLSTMCell(256,256,kernel_size=3, padding=1, stride=1)
            self.convRNN2 = ConvLSTMCell(256,128,kernel_size=3, padding=1, stride=1)
        elif r_type == "gru":
            self.convRNN1 = ConvGRUCell(256,256,3)
            self.convRNN2 = ConvGRUCell(256,128,3)
        else:
            print("Error : r_type")
            return -1
        
        """
        self.classifier = nn.Sequential(
            nn.Conv2d(128,6,kernel_size=1, padding=0, stride=1),
            nn.AvgPool2d(kernel_size=6, stride=1, padding=0)
        )
        """
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 6)
        )
        
        if copyParameters:
            self.copyParameters(models.alexnet(pretrained=True))
        
        
    def copyParameters(self, alexnetModel):
        for i, a in enumerate(alexnetModel.features):
            if type(a) is torch.nn.modules.conv.Conv2d:
                self.features[i].weight = a.weight
                self.features[i].bias   = a.bias
    
    def cuda(self, device=None):
        self.isCuda = True
        return self._apply(lambda t: t.cuda(device))
    
    def forward(self, x):
        batchSize = len(x[0])
        if self.r_type=="lstm":
            outputs = []
            
            if self.isCuda:
                ht1 = Variable( torch.zeros(batchSize,256,6,6)).cuda()
                ct1 = Variable( torch.zeros(batchSize,256,6,6)).cuda()
                ht2 = Variable( torch.zeros(batchSize,128,6,6)).cuda()
                ct2 = Variable( torch.zeros(batchSize,128,6,6)).cuda()
            else:
                ht1 = Variable( torch.zeros(batchSize,256,6,6))
                ct1 = Variable( torch.zeros(batchSize,256,6,6))
                ht2 = Variable( torch.zeros(batchSize,128,6,6))
                ct2 = Variable( torch.zeros(batchSize,128,6,6))
                
            for i in x:
                xt = self.features(i)
                o, (ht1,ct1) = self.convRNN1(xt, (ht1, ct1))
                o, (ht2,ct2) = self.convRNN2(o, (ht2, ct2))
                outputs.append(o)
        
        elif self.r_type=="gru":
            outputs = []
            if self.isCuda:
                ht1 = Variable( torch.zeros(batchSize,256,6,6)).cuda()
                ht2 = Variable( torch.zeros(batchSize,128,6,6)).cuda()
            else:
                ht1 = Variable( torch.zeros(batchSize,256,6,6))
                ht2 = Variable( torch.zeros(batchSize,128,6,6))
            
            for e,i in enumerate(x):
                xt = self.features(i)
                ht1 = self.convRNN(xt, ht1)
                ht2 = self.convRNN(ht1, ht2)
                outputs.append(ht2)
        return self.classifier(outputs[-1].view(batchSize, -1))
        #x = self.classifier(x).squeeze().unsqueeze(0)


# In[239]:

if __name__=="__main__":
    testconvLSTMCell()
    testGruCell()
    x = Variable(torch.Tensor(3,2,3,225,225)).cuda()
    m = convRNN_1_layer("gru").cuda()
    print(m(x).size() == torch.Size([2,6]))
    x = Variable(torch.Tensor(3,2,3,225,225)).cuda()
    m = convRNN_2_layer("lstm").cuda()
    print(m(x).size() == torch.Size([2, 6]))
    


# In[240]:




# In[309]:

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    
class ResNet_lstm(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 8, 36, 3]):
        self.block = block
        
        self.inplanes = 64
        super(ResNet_lstm, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.isCuda = False
        self.convRNN1 = ConvLSTMCell(512 * block.expansion, 512 * block.expansion,kernel_size=3, padding=1, stride=1)
        self.convRNN2 = ConvLSTMCell(512 * block.expansion, 512 * block.expansion,kernel_size=3, padding=1, stride=1)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, 6)
        self.fc = nn.Linear(2048, 6)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def cuda(self, device=None):
        self.isCuda = True
        return self._apply(lambda t: t.cuda(device))
    
    def forward(self, x):
        batchSize = x.size(1)
        if self.isCuda:
            ht1 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8)).cuda()
            ct1 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8)).cuda()
            ht2 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8)).cuda()
            ct2 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8)).cuda()
        else:
            ht1 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8))
            ct1 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8))
            ht2 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8))
            ct2 = Variable( torch.zeros(batchSize, 512 * self.block.expansion,8,8))
        outputs = []
        for i in x:
            xt = self.conv1(i)
            xt = self.bn1(xt)
            xt = self.relu(xt)
            xt = self.maxpool(xt)

            xt = self.layer1(xt)
            xt = self.layer2(xt)
            xt = self.layer3(xt)
            xt = self.layer4(xt)
            
            o, (ht1,ct1) = self.convRNN1(xt, (ht1, ct1))
            o, (ht2,ct2) = self.convRNN2(o, (ht2, ct2))
            outputs.append(o)

        o = self.avgpool(outputs[-1])
        o = o.view(batchSize, -1)
        o = self.fc(o)
        return o


# In[310]:

if __name__=="__main__":
    x = Variable(torch.Tensor(3,2,3,225,225)).cuda()
    m = ResNet_lstm(BasicBlock, [2, 2, 2, 2]).cuda()
    print(m(x).size() == torch.Size([2, 6]))

