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
    pred_lbls = []
    pred_ious = []
    pred_boxs_x1 = []
    pred_boxs_y1 = []
    pred_boxs_x2 = []
    pred_boxs_y2 = []    
    true_lbls = []
    true_boxs_x1 = []
    true_boxs_y1 = []
    true_boxs_x2 = []
    true_boxs_y2 = []
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
        crm = output
                          
        # Generate keypoints
        # # # # # # # # # # #
        output = processor.fields(data)
        keypoint_sets, scores = processor.keypoint_sets(output[0])
        
        # Get the bounding boxes and their labels from the keypoints and the crm map
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        bboxes_pred_0, bboxes_pred_1 = [], []
        for i,keypoint_set in enumerate(keypoint_sets):
        
            # bbox of current keypoint
            x_ = int(np.amin(keypoint_set[:,0]))
            y_ = int(np.amin(keypoint_set[:,1]))
            w_ = int(np.amax(keypoint_set[:,0])) - x_
            h_ = int(np.amax(keypoint_set[:,1])) - y_
            
            # generate the distribution centered at this box
            x0, y0, sigma_x, sigma_y = x_+float(w_)/2, y_+float(h_)/2, float(w_)/4, float(h_)/4
            
            # activity map for current person
            y, x = np.arange(378), np.arange(960)    
            gy = np.exp(-(y-y0)**2/(2*sigma_y**2))
            gx = np.exp(-(x-x0)**2/(2*sigma_x**2))
            g  = np.outer(gy, gx)
                        
            state = np.argmax( [np.sum(g*crm[:,:,0]), np.sum(g*crm[:,:,1])] )
            if(state==0):
                bboxes_pred_0.append(np.array((x_,y_,x_+w_,y_+h_)))
            if(state==1):
                bboxes_pred_1.append(np.array((x_,y_,x_+w_,y_+h_)))

        # determine if the detected bounding boxes coincide with the ground truth   
        # those whose ground truth are occluded should have been removed     
        for x,y,w,h,label,pedestrian,frame in zip(meta[0]["box_x"], meta[0]["box_y"], meta[0]["box_w"], meta[0]["box_h"], meta[0]["labels"], meta[0]["pedestrian"], meta[0]["frame"]):
        
            bbox_true = np.array((x,y,x+w,y+h))
        
            # compute the ious
            iou_list_0 = [iou(bbox_true, bbox_pred_0) for bbox_pred_0 in bboxes_pred_0]
            iou_list_1 = [iou(bbox_true, bbox_pred_1) for bbox_pred_1 in bboxes_pred_1]

            # compute best iou
            max_iou_0 = 0.0
            max_iou_1 = 0.0
            if(len(iou_list_0) == 0):
                max_iou_score_0 = 0.0
            else:
                max_iou_ind_0   = np.argmax(iou_list_0)
                max_iou_score_0 = iou_list_0[max_iou_ind_0]
                
            if(len(iou_list_1) == 0):
                max_iou_score_1 = 0.0
            else:
                max_iou_ind_1   = np.argmax(iou_list_1)
                max_iou_score_1 = iou_list_1[max_iou_ind_1]
                
            # assign bbox
            # negative
            if(max_iou_score_0 > max_iou_score_1):                
                best_bbox = bboxes_pred_0[max_iou_ind_0]
                pred_lbls.append(0)
                pred_ious.append(max_iou_score_0)
                pred_boxs_x1.append(best_bbox[0])
                pred_boxs_y1.append(best_bbox[1])
                pred_boxs_x2.append(best_bbox[2])
                pred_boxs_y2.append(best_bbox[3]) 
            # positive               
            if(max_iou_score_1 > max_iou_score_0):                
                best_bbox = bboxes_pred_1[max_iou_ind_1]
                pred_lbls.append(1)
                pred_ious.append(max_iou_score_1)
                pred_boxs_x1.append(best_bbox[0])
                pred_boxs_y1.append(best_bbox[1])
                pred_boxs_x2.append(best_bbox[2])
                pred_boxs_y2.append(best_bbox[3])
            # miss
            if((max_iou_score_0 == 0.0 and max_iou_score_1 == 0.0) or (max_iou_score_0 == max_iou_score_1)):
                pred_lbls.append(-1)
                pred_ious.append(0.0)
                pred_boxs_x1.append(-1)
                pred_boxs_y1.append(-1)
                pred_boxs_x2.append(-1)
                pred_boxs_y2.append(-1)
                
            # put in gt    
            true_lbls.append(label)
            true_boxs_x1.append(x)
            true_boxs_y1.append(y)
            true_boxs_x2.append(x+w)
            true_boxs_y2.append(y+h)
            filepaths.append(meta[0]["path_to_scene"])
            pedestrians.append(pedestrian)
            frames.append(frame)

        ############################################################################################
         
    df["pred_lbl"] = pred_lbls
    df["pred_iou"] = pred_ious
    df["pred_box_x1"] = pred_boxs_x1 
    df["pred_box_y1"] = pred_boxs_y1 
    df["pred_box_x2"] = pred_boxs_x2 
    df["pred_box_y2"] = pred_boxs_y2 
    df["true_lbl"] = true_lbls
    df["true_box_x1"] = true_boxs_x1 
    df["true_box_y1"] = true_boxs_y1 
    df["true_box_x2"] = true_boxs_x2 
    df["true_box_y2"] = true_boxs_y2 
    df["path_to_scene"] = filepaths
    df["pedestrian"] = pedestrians
    df["frames"] = frames
    df.to_csv(args.result, index=False)

if __name__ == '__main__':
    main()
