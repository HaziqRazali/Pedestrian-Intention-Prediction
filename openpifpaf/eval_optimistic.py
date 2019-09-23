"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import sys
import os

import pandas as pd
import numpy as np
import torch
import cv2

from .network import nets
from . import datasets, decoder, show

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_score = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou_score

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
    parser.add_argument('--threshold', default=0, type=int,
                        help='threshold')  
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
    
    # data loader
    # do not ignore last
    _, _, _, \
    _, jaad_val_loader, _ = datasets.train_factory(args, preprocess=[], target_transforms=[], jaad_datasets=[args.jaad_train, args.jaad_val, args.jaad_pre_train])
    
    # loop through validation set        
    pred = []
    true = []
    filepaths = []
    pedestrians = []
    frames = []
    df = pd.DataFrame()
    for data, targets, meta in jaad_val_loader:
    
        print(meta[0]["path_to_scene"])
    
        if args.device:
            data = data.to(args.device, non_blocking=True)
            targets = [[t.to(args.device, non_blocking=True) for t in head] for head in targets]

        # Generate distribution
        # # # # # # # # # # #
        output = model(data, head="crm")
        output = output[0][0][0].cpu().detach().numpy()
        output = np.transpose(output, [1,2,0])
        output = output * 255
        output = cv2.resize(output, (960,378))
        output[output < 50] = 0
        crm = output
                                
        # determine if the detected bounding boxes coincide with the ground truth  
        # !!!!! for classification   
        for x,y,w,h,label,pedestrian,frame in zip(meta[0]["box_x"], meta[0]["box_y"],meta[0]["box_w"],meta[0]["box_h"],meta[0]["labels"],meta[0]["pedestrian"],meta[0]["frame"]):
        
            ## evaluate on subset of humans that have been detected by pifpaf
            #with open(os.path.join(pedestrian.replace("image","keypoints"),frame+".pifpaf.json")) as json_file:   
            #    data = json.load(json_file)  
            #    kps_curr_ped = np.array([d["keypoints"] for d in data])
            #    kps_curr_ped = np.reshape(kps_curr_ped, (np.shape(kps_curr_ped)[0],17,3))                
            #    # no keypoints
            #    if(np.shape(kps_curr_ped)[0]==0):
            #        continue
                           
            score0 = np.sum(crm[y:y+h,x:x+w,0])
            score1 = np.sum(crm[y:y+h,x:x+w,1])            
            if(score1 + score0 > 0):          
                pred.append(score1/(score1+score0))
                true.append(label)
            else:
                pred.append(0)
                true.append(label)
            filepaths.append(meta[0]["path_to_scene"])
            pedestrians.append(pedestrian)
            frames.append(frame)
            
    df["pred"] = pred
    df["true"] = true
    df["filepath"] = filepaths
    df["pedestrian"] = pedestrians
    df["frame"] = frames
    df.to_csv(args.result,index=False)

if __name__ == '__main__':
    main()
