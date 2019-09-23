"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import sys
import os
from PIL import Image

import torchvision
import numpy as np
import torch
import cv2

from .network import nets
from . import datasets, decoder, show


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
    datasets.train_cli(parser)
    decoder.cli(parser, force_complete_pose=True, instance_threshold=0.2)
    
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
    parser.add_argument('--result', default="results.txt", type=str,help='result')
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

def main():

    args = cli()
    
    # load model
    model, _ = nets.factory_from_args(args)
    model = model.to(args.device)
    model = model.eval()
    processor = decoder.factory_from_args(args, model)
    
    # painter
    skeleton_painter = show.KeypointPainter(show_box=False, color_connections=False, markersize=0.5, linewidth=1)
    
    # data loader  
    _, _, _, \
    _, jaad_val_loader, _ = datasets.train_factory(args, preprocess=[], target_transforms=[], jaad_datasets=[args.jaad_train, args.jaad_val, args.jaad_pre_train])
    
    # !!!! make sure video resolution is same as keypoints and crm resolution
    vid_w = 960
    vid_h = 378
    vid = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, (vid_w,vid_h))
    for data, targets, meta in jaad_val_loader:
    
        print(meta[0]["path_to_scene"])
        
        if args.device:
            data = data.to(args.device, non_blocking=True)
        
        # Generate distribution
        # # # # # # # # # # #
        output = model(data, head="crm")
        output = output[0][0][0].cpu().detach().numpy()
        output = np.transpose(output, [1,2,0])
        output = output * 255
        #output = output.astype(np.uint8) convert later
        
        # resize output to resnet input size
        output = cv2.resize(output, (vid_w,vid_h))
        fields0 = output[:,:,0]
        fields1 = output[:,:,1] 
                              
        # set the color of fields 0 to be green
        fields0 = np.repeat(fields0[:, :, np.newaxis], 3, axis=2)
        fields0[:,:,0] = 0
        fields0[:,:,2] = 0            
        # set the color of fields 1 to be red
        fields1 = np.repeat(fields1[:, :, np.newaxis], 3, axis=2)
        fields1[:,:,0] = 0
        fields1[:,:,1] = 0        
        im = cv2.imread(meta[0]["path_to_scene"])
        im = cv2.resize(im, (960, 540))
        im = im[162:,:,:]
        im = im.astype(np.float) 
        im = im + fields0 + fields1
        im[im > 255] = 255
        im = im.astype(np.uint8) 
        # convert bgr to rgb
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        im = torchvision.transforms.functional.to_tensor(im)
        im = im.permute(1, 2, 0)
                          
        # Generate keypoints
        # # # # # # # # # # #
        pifpaf_output = processor.fields(data)
        keypoint_sets, scores = processor.keypoint_sets(pifpaf_output[0])
        
        # get the bounding box of each keypoint set
        colors = []
        activity_map = np.zeros((im.size(0), im.size(1)),dtype=np.float32)
        for i,keypoint_set in enumerate(keypoint_sets):
            x = int(np.amin(keypoint_set[:,0]))
            y = int(np.amin(keypoint_set[:,1]))
            w = int(np.amax(keypoint_set[:,0])) - x
            h = int(np.amax(keypoint_set[:,1])) - y
            
            # generate the distribution centered at this box
            x0, y0, sigma_x, sigma_y = x+float(w)/2, y+float(h)/2, float(w)/4, float(h)/4
            
            # activity map for current person
            y, x = np.arange(im.size(0)), np.arange(im.size(1))    
            gy = np.exp(-(y-y0)**2/(2*sigma_y**2))
            gx = np.exp(-(x-x0)**2/(2*sigma_x**2))
            g  = np.outer(gy, gx)
                        
            state = np.argmax( [np.sum(g*output[:,:,0]), np.sum(g*output[:,:,1])] )
            if(state==0):
                colors.append("green")
            if(state==1):
                colors.append("red")
        
        with show.image_canvas(im, 'im.png', show=args.show, fig_width=10.0, dpi_factor=1.0) as ax: skeleton_painter.keypoints(ax, keypoint_sets, scores=scores, colors=colors)
        # write to video
        im = cv2.imread("im.png")
        vid.write(im)

if __name__ == '__main__':
    main()
