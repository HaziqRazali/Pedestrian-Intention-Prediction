"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import pandas as pd
import numpy as np
import torch
import cv2
import gc

import torchvision.models as models
from torch.autograd import Variable
from torch.nn import ReLU
from .network import nets
from . import datasets, decoder, show

#def get_bboxes(im, dy=7.875, dx=8.0):
#    
#    #im = im * 255
#    
#    # get the bboxes of all those not crossing
#    bboxes = []    
#    im = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)[1]
#    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im, 8, cv2.CV_32S)
#    for x,y,w,h,_ in stats:
#        if(h == np.shape(im)[0] or w == np.shape(im)[1]):
#            continue
#        else:
#            x = x * dx
#            y = y * dy
#            w = w * dx
#            h = h * dy
#            bboxes.append(np.array(( int(np.floor(x)), int(np.floor(y)), int(np.ceil(x+w)), int(np.ceil(y+h)) )))
#    return bboxes
    
def get_bboxes(im):
    
    im = im.detach().cpu().numpy()
    im = im * 255
    im = im.astype(np.uint8)
    
    # get the bboxes of all those not crossing
    bboxes = []    
    im = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im, 8, cv2.CV_32S)
    for x,y,w,h,_ in stats:
        if(h == np.shape(im)[0] or w == np.shape(im)[1]):
            continue
        else:
            bboxes.append(np.array(( int(np.floor(x)), int(np.floor(y)), int(np.ceil(x+w)), int(np.ceil(y+h)) )))
    return bboxes

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    datasets.train_cli(parser)
    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--output-directory',
                        help=('Output directory. When using this option, make '
                              'sure input images have distinct file names.'))
    parser.add_argument('--show', default=False, action='store_true',
                        help='show image of output overlay')
    parser.add_argument('--output-types', nargs='+', default=['skeleton', 'json'],
                        help='what to output: skeleton, keypoints, json')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--partial', default=0, type=int, help='figure width') 
    parser.add_argument('--figure-width', default=10.0, type=float,
                        help='figure width')
    parser.add_argument('--dpi-factor', default=1.0, type=float,
                        help='increase dpi of output image by this factor')
    group = parser.add_argument_group('logging')
    group.add_argument('-q', '--quiet', default=False, action='store_true',
                       help='only show warning messages or above')
    group.add_argument('--debug', default=False, action='store_true',
                       help='print debug messages')
    args = parser.parse_args()

    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    return args

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
                        
        self.update_relus()
        self.hook_layers()
        
    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            #print(self.gradients)
            #print("grad in ", len(grad_in))
            #print(grad_in[0].size())
            #print(grad_in[1].size())
            #print(grad_in[2].size())
        # Register hook to the first layer
        for module in self.model.base_net.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.register_backward_hook(hook_function)
                break

    def update_relus(self):        
    
        # relu backward pass for guided backprop
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)
                
        # relu forward pass
        def relu_forward_hook_function(module, inp, out):
            self.forward_relu_outputs.append(out)
                
        # Loop through convolutional feature extractor and hook up ReLUs with relu_hook_function                
        for module in self.model.base_net.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(relu_forward_hook_function)
                module.register_backward_hook(relu_backward_hook_function)

    def generate_gradients(self, input_image, keep, suppress):
                
        # Forward pass
        input_image = Variable(input_image, requires_grad=True)
        model_output = self.model(input_image, head="crm")[0][0].squeeze()
        
        #print(model_output.size())
        for i in range(47):
            for j in range(119):
                a=model_output[0,i,j]
                b=model_output[1,i,j]
                #print("before suppression ", model_output[0,i,j], model_output[1,i,j])
                if(a>b):
                    model_output[1,i,j] = 0
                if(b>a):
                    model_output[0,i,j] = 0
                #print("after suppression ", model_output[0,i,j], model_output[1,i,j])
                #input()
                                                  
        # sizes
        channels, height, width = model_output.size()
                               
        # backward pass for non crosser 
        model_output_0 = model_output.clone()
        model_output_0[keep][model_output[keep,:,:] > 0] = 1
        model_output_0[suppress] = 0
        model_output_0 = model_output_0.unsqueeze(0)
        self.model.zero_grad()
        model_output.backward(gradient=model_output_0)
        gradients_as_arr0 = self.gradients.data.cpu().numpy()[0]
        gradients_as_arr0 = np.transpose(gradients_as_arr0, [1,2,0])
               
        #del model_output, model_output_0, input_image
        return gradients_as_arr0
                
    def generate_gradients_box(self, input_image, keep, x, y, w, h):
                
        # Forward pass
        input_image = Variable(input_image, requires_grad=True) 
        model_output = self.model(input_image, head="crm")[0][0].squeeze()
                                                
        # backward pass for non crosser 
        #model_output_0 = model_output.clone()
        roi = torch.cuda.FloatTensor(model_output.size()).zero_()
        roi[keep,y:y+h,x:x+w] = 1
        roi = roi.unsqueeze(0)
                
        #print(model_output_0.size())
        #print(y,y+h,x,x+w)
        
        #model_output_0[:,:,:] = 0
        #print(model_output_0.size())
        
        #model_output_0[keep,y:y+h,x:x+w] = 1
        #model_output_0[suppress] = 0
        #model_output_0 = model_output_0.unsqueeze(0)
        self.model.zero_grad()
        model_output.backward(gradient=roi)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        gradients_as_arr = np.transpose(gradients_as_arr, [1,2,0])
               
        return gradients_as_arr

def main():

    args = cli()
    
    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    model = model.eval()
    
    # data loader
    _, _, _, \
    _, jaad_val_loader, _ = datasets.train_factory(args, preprocess=[], target_transforms=[], jaad_datasets=[args.jaad_train, args.jaad_val, args.jaad_pre_train])
    
    # Set up guided backprop
    GBP = GuidedBackprop(model)
    
    # paint gbp
    grad0 = cv2.VideoWriter("grad0.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, (960,378))
    grad1 = cv2.VideoWriter("grad1.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, (960,378))
        
    ## NEW
    if(0):
        val = 127
        for data, targets, meta in jaad_val_loader:
        
            print(meta[0]["path_to_scene"])
        
            if args.device:
                data = data.to(args.device, non_blocking=True) 
        
            guided_grads0_list = []
            guided_grads1_list = []
            # in case there are no pedestrians for the current scene
            guided_grads0 = np.zeros((378,960,3),dtype=np.float32)
            guided_grads1 = np.zeros((378,960,3),dtype=np.float32)
            for x,y,w,h,label in zip(meta[0]["box_x"], meta[0]["box_y"],meta[0]["box_w"],meta[0]["box_h"],meta[0]["labels"]):
                
                x = int(np.floor(float(x)/8))
                y = int(np.floor(float(y)/8))
                w = int(np.ceil(float(w)/8))
                h = int(np.ceil(float(h)/8))            
                         
                # initialize for the current pedestrian
                guided_grads0 = np.zeros((378,960,3),dtype=np.float32)
                guided_grads1 = np.zeros((378,960,3),dtype=np.float32)
                if(label == 0):
                    guided_grads0 = GBP.generate_gradients_box(data, 0, x,y,w,h)
                if(label == 1):
                    guided_grads1 = GBP.generate_gradients_box(data, 1, x,y,w,h)  
                                                        
                guided_grads0_list.append(guided_grads0)
                guided_grads1_list.append(guided_grads1)
            # overwrite above if there are pedestrian
            # need to absolute first
            if(len(guided_grads0_list)!=0):   
                guided_grads0 = np.sum(guided_grads0_list,0)  
            if(len(guided_grads1_list)!=0):   
                guided_grads1 = np.sum(guided_grads1_list,0)  
            
            if(np.amin(guided_grads0) != 0):           
                guided_grads0 = ((guided_grads0 - np.amin(guided_grads0)) / (np.amax(guided_grads0) - np.amin(guided_grads0))) * 255
                guided_grads0_median = np.median(guided_grads0)
                guided_grads0 = guided_grads0 + (val-guided_grads0_median)
                guided_grads0[guided_grads0 < 0] = 0
                guided_grads0[guided_grads0 > 255] = 255
                guided_grads0 = guided_grads0.astype(np.uint8)
            else:                    
                guided_grads0.fill(val)
                guided_grads0 = guided_grads0.astype(np.uint8)
            if(np.amin(guided_grads1) != 0):
                guided_grads1 = ((guided_grads1 - np.amin(guided_grads1)) / (np.amax(guided_grads1) - np.amin(guided_grads1))) * 255
                guided_grads1_median = np.median(guided_grads1)
                guided_grads1 = guided_grads1 + (val-guided_grads1_median)
                guided_grads1[guided_grads1 < 0] = 0
                guided_grads1[guided_grads1 > 255] = 255
                guided_grads1 = guided_grads1.astype(np.uint8)    
            else:                    
                guided_grads1.fill(val)
                guided_grads1 = guided_grads1.astype(np.uint8) 
               
            grad0.write(guided_grads0)
            grad1.write(guided_grads1)
        
    # OLD
    # -----------------------------------------
    if(1):
        for data, targets, meta in jaad_val_loader:
        
            if args.device:
                data = data.to(args.device, non_blocking=True)
        
            print(meta[0]["path_to_scene"])
            # Generate gradients
            # # # # # # # # # # #
            guided_grads0 = GBP.generate_gradients(data, 0, 1)
            guided_grads1 = GBP.generate_gradients(data, 1, 0)  
            val = 155    
            if(1):
                if(np.amin(guided_grads0) != 0):        
                    guided_grads0 = ((guided_grads0 - np.amin(guided_grads0)) / (np.amax(guided_grads0) - np.amin(guided_grads0))) * 255
                    guided_grads0_median = np.median(guided_grads0)
                    guided_grads0 = guided_grads0 + (val-guided_grads0_median)
                    guided_grads0[guided_grads0 < 0] = 0
                    guided_grads0[guided_grads0 > 255] = 255
                    guided_grads0 = guided_grads0.astype(np.uint8)
                else:
                    guided_grads0.fill(val)
                if(np.amin(guided_grads1) != 0):
                    guided_grads1 = ((guided_grads1 - np.amin(guided_grads1)) / (np.amax(guided_grads1) - np.amin(guided_grads1))) * 255
                    guided_grads1_median = np.median(guided_grads1)
                    guided_grads1 = guided_grads1 + (val-guided_grads1_median)
                    guided_grads1[guided_grads1 < 0] = 0
                    guided_grads1[guided_grads1 > 255] = 255
                    guided_grads1 = guided_grads1.astype(np.uint8)
                else:
                    guided_grads1.fill(val)
                       
            grad0.write(guided_grads0)
            grad1.write(guided_grads1)

if __name__ == '__main__':
    main()
