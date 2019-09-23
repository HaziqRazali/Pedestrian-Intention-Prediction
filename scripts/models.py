import sys
import pickle

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import math

from torch.autograd import Variable
from torch.nn import ReLU
from collections import OrderedDict

class CNNLSTM(nn.Module):
    def __init__(self, args, grad=False):
        super(CNNLSTM, self).__init__()

        self.lstm_h_dim = args.lstm_h_dim
        self.dropout = args.dropout

        # gradients
        self.gradients = None

        # Basenet
        #self.model = models.resnet50(pretrained=True)
        #self.model = nn.Sequential(*list(self.model.children())[0])
        self.basenet = models.resnet50(pretrained=False)                
        self.basenet = torch.nn.Sequential(*(list(self.basenet.children())[:-2]))  
        self.basenet.load_state_dict(torch.load("./models/pifpaf-resnet50.pt"))             
        for param in self.basenet.parameters():
            param.requires_grad = True
        self.pool = nn.AdaptiveAvgPool2d((1,1))
      
        # LSTM
        self.lstm = nn.LSTM(2048, args.lstm_h_dim, 1, batch_first=False)

        # Linear classifier
        self.linear_classifier = nn.Linear(args.lstm_h_dim, 2)

        # CNN gradients disabled
        if(grad==False):        
            for name, param in self.basenet.named_parameters():         
                if param.requires_grad:
                    param.requires_grad = False  
        # CNN gradients enabled for guided backprop        
        else: 
            for name, param in self.basenet.named_parameters():         
                if param.requires_grad:
                    param.requires_grad = True       
                self.hook_layers()
                self.update_relus()
        
    # Set hook to the first CNN layer
    def hook_layers(self):    
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            # grad_out[0].size = timesteps x 64 x 100 x 40 
            # grad_in[0].size = None
            # grad_in[1].size = 64 x 3 x 3 x 3 (64 3x3x3 filters)
            # grad_in[2].size = 64 (biases)
        self.model[0].register_backward_hook(hook_function)

    # Updates relu activation functions so that it only returns positive gradients
    def update_relus(self):
        # Clamp negative gradients to 0
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through convolutional feature extractor and hook up ReLUs with relu_hook_function
        for i in range(len(self.model)):
            if isinstance(self.model[i], ReLU):
                self.model[i].register_backward_hook(relu_hook_function)
        # Loop through MLP and hook up ReLUs with relu_hook_function
        for i in range(len(self.classifier)):
            if isinstance(self.classifier[i], ReLU):
                self.classifier[i].register_backward_hook(relu_hook_function)

    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.lstm_h_dim).cuda(),
            torch.zeros(1, batch, self.lstm_h_dim).cuda()
        )

    def forward(self, images_pedestrian_all, keypoints_pedestrian_all, input_as_var = False, classify_every_timestep=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """

        # batch = number of pedestrians where sequence length of each pedestrian varies 
        batch, length, channels, height, width = images_pedestrian_all.size()
        
        # extract features
        images_pedestrian_all = images_pedestrian_all.cuda()        
        images_pedestrian_all = images_pedestrian_all.view(batch*length,channels,height,width)
        images_pedestrian_all = self.pool(self.basenet(images_pedestrian_all)).squeeze() # 512, 2048
        images_pedestrian_all = images_pedestrian_all.view(batch,length,images_pedestrian_all.size(1))
        images_pedestrian_all = images_pedestrian_all.permute(1,0,2)
        
        # send through lstm and get the final output
        state_tuple = self.init_hidden(1)
        output, state = self.lstm(images_pedestrian_all)
        y_pred = self.linear_classifier(state[0].squeeze())

        ## for each pedestrian
        ##features_pedestrian_all = []
        #state_all = []
        #for images_pedestrian_i in images_pedestrian_all:
        #           
        #        # sequence length
        #        seq_len = images_pedestrian_i.size(0)
        #                  
        #        # if we want the input to be a Variable
        #        # used for guided backprop
        #        if(input_as_var == True):
        #            images_pedestrian_i = Variable(images_pedestrian_i.cuda(), requires_grad=True)
        #        else:
        #            images_pedestrian_i = images_pedestrian_i.cuda()
        #
        #        # send all the images of the current pedestrian through the CNN feature extractor
        #        features_pedestrian_i = self.basenet(images_pedestrian_i)
        #        features_pedestrian_i = self.pool(features_pedestrian_i)
        #        features_pedestrian_i = features_pedestrian_i.view(seq_len, -1)

        #        # embed the features
        #        features_pedestrian_i = F.dropout(features_pedestrian_i, p=self.dropout)
        #        features_pedestrian_i = torch.unsqueeze(features_pedestrian_i, 1)

        #        # send through lstm and get the final output
        #        state_tuple = self.init_hidden(1)
        #        output, state = self.lstm(features_pedestrian_i)       
        #        state_all.append(F.relu(F.dropout(state[0].squeeze(), p=self.dropout)))

        #state_all = torch.stack(state_all, dim=0)
        #y_pred = self.linear_classifier(state_all)
        return y_pred
        
######################################################
class LSTMKP(nn.Module):
    def __init__(self, args, grad=False):
        super(LSTMKP, self).__init__()

        self.lstm_h_dim = args.lstm_h_dim
        self.dropout = args.dropout

        # gradients
        self.gradients = None

        # Basenet
        self.linear_embedder = nn.Linear(34, 1024)
      
        # LSTM
        self.lstm = nn.LSTM(1024, args.lstm_h_dim, 1, batch_first=False)

        # Linear classifier
        self.linear_classifier = nn.Linear(args.lstm_h_dim, 2)

    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.lstm_h_dim).cuda(),
            torch.zeros(1, batch, self.lstm_h_dim).cuda()
        )

    def forward(self, images_pedestrian_all, keypoints_pedestrian_all, input_as_var = False, classify_every_timestep=False):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        
        #print(images_pedestrian_all.size(), keypoints_pedestrian_all.size()) # 32,16,34
        #sys.exit(0)

        # batch = number of pedestrians where sequence length of each pedestrian varies 
        batch, length, feature_dim = keypoints_pedestrian_all.size()
        
        # extract features
        keypoints_pedestrian_all = keypoints_pedestrian_all.cuda()        
        keypoints_pedestrian_all = self.linear_embedder(keypoints_pedestrian_all) # 512, 2048
        keypoints_pedestrian_all = keypoints_pedestrian_all.permute(1,0,2)
        
        # send through lstm and get the final output
        state_tuple = self.init_hidden(1)
        output, state = self.lstm(keypoints_pedestrian_all)
        y_pred = self.linear_classifier(state[0].squeeze())

        return y_pred

        
###############################################
## 3D RESNET                                  #
###############################################
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
                
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        # i should simply use adaptive pool
        # ---------------------------------
        #last_duration = int(math.ceil(sample_duration / 16))
        #last_size = int(math.ceil(sample_size / 32))
        #self.avgpool = nn.AvgPool3d((1, 4, 2), stride=1)
        # i should simply use adaptive pool
        # ---------------------------------
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        #self.fc1 = nn.Linear(512 * block.expansion, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        
        x = x.permute(0, 2, 1, 3, 4).cuda()
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)

        #x = x.view(x.size(0), -1)
        x = x.squeeze()
        #x = self.fc1(x)
        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def CNN3D(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

###############################################
## DENSENET                                   #
###############################################

def densenet121(**kwargs):
    model = DenseNet(
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        **kwargs)
    return model
    
def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 sample_size,
                 sample_duration,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):

        super(DenseNet, self).__init__()

        self.sample_size = sample_size
        self.sample_duration = sample_duration

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ('conv0',
                 nn.Conv3d(
                     3,
                     num_init_features,
                     kernel_size=7,
                     stride=(1, 2, 2),
                     padding=(3, 3, 3),
                     bias=False)),
                ('norm0', nn.BatchNorm3d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)
        self.classifier1 = nn.Linear(num_features, 2)

    def forward(self, x, y):
        x = x.permute(0, 2, 1, 3, 4).cuda()
        features = self.features(x)
        out = F.relu(features, inplace=True)
        last_duration = int(math.ceil(self.sample_duration / 16))
        last_size = int(math.floor(self.sample_size / 32))
        out = F.avg_pool3d(
            out, kernel_size=(1, 3, 1)).view(
                features.size(0), -1)
        #out = self.classifier(out)
        out = self.classifier1(out)
        return out