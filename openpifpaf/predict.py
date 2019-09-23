"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from .network import nets
from . import datasets, decoder, show


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    nets.cli(parser)
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
    parser.add_argument('--loader-workers', default=2, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
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

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

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
    processor = decoder.factory_from_args(args, model)

    # data
    data = datasets.ImageList(args.images)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False,
        pin_memory=args.pin_memory, num_workers=1)

    ## visualizers
    keypoint_painter = show.KeypointPainter(show_box=False)
    skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                            markersize=1, linewidth=1)

    for image_i, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)

        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        
        pifpaf_batch = processor.fields(processed_images)
        
        # Get the activity map
        # Resolution of 540X960 seems to work best
        # =============================================
        crm_batch = model(processed_images, head="crm")
        # =============================================

        # unbatch
        #for image_path, image, processed_image_cpu, pifpaf in zip(image_paths, images, processed_images_cpu, pifpaf_batch):
        for image_path, image, processed_image_cpu, pifpaf, crm in zip(image_paths, images, processed_images_cpu, pifpaf_batch, crm_batch):

            if args.output_directory is None:
                output_path = image_path
            else:
                file_name = os.path.basename(image_path)
                output_path = os.path.join(args.output_directory, file_name)
            logging.info('image %d: %s to %s', image_i, image_path, output_path)

            print(output_path)              
            ######################################
            # CRM                                #
            ######################################
            crm = crm[0][0].detach().cpu().numpy()
            print(np.amin(crm), np.max(crm))
            #crm = crm * 255
            #crm = crm.astype(np.uint8)
            crm = np.transpose(crm, [1,2,0])
            crm = cv2.resize(crm, (image.size(1), image.size(0)))
            crm0 = crm[:,:,0]
            crm1 = crm[:,:,1]            
            # set the color of fields 1 to be green
            crm0 = np.repeat(crm0[:, :, np.newaxis], 3, axis=2)
            crm0[:,:,0] = 0
            crm0[:,:,2] = 0   
            crm0 = torch.FloatTensor(crm0)         
            # set the color of fields 1 to be blue
            crm1 = np.repeat(crm1[:, :, np.newaxis], 3, axis=2)
            crm1[:,:,1] = 0
            crm1[:,:,2] = 0
            crm1 = torch.FloatTensor(crm1)
            # paint crm on image
            print(torch.max(image), torch.max(crm0), torch.max(crm1))
            print(type(image), type(crm0), type(crm1))
            image = image + crm0 + crm1
            image[image > 1] = 1
                                            
            processor.set_cpu_image(image, processed_image_cpu)
            keypoint_sets, scores = processor.keypoint_sets(pifpaf)
            keypoint_sets = keypoint_sets * 2
            colors = []*len(keypoint_sets)
                                    
            # get the bounding box of each keypoint set
            activity_map = np.zeros((image.size(0), image.size(1)),dtype=np.float32)
            for i,keypoint_set in enumerate(keypoint_sets):
                x = int(np.amin(keypoint_set[:,0]))
                y = int(np.amin(keypoint_set[:,1]))
                w = int(np.amax(keypoint_set[:,0])) - x
                h = int(np.amax(keypoint_set[:,1])) - y
                
                # generate the distribution centered at this box
                x0, y0, sigma_x, sigma_y = x+float(w)/2, y+float(h)/2, float(w)/4, float(h)/4
                
                # activity map for current person
                y, x = np.arange(image.size(0)), np.arange(image.size(1))    
                gy = np.exp(-(y-y0)**2/(2*sigma_y**2))
                gx = np.exp(-(x-x0)**2/(2*sigma_x**2))
                g  = np.outer(gy, gx)
                
                state = np.argmax( [np.sum(g*crm[:,:,0]), np.sum(g*crm[:,:,1])] )
                if(state==0):
                    colors.append("green")
                if(state==1):
                    colors.append("red")             
                #activity_map += g
            
            print(image.size())
            if 'skeleton' in args.output_types:
                with show.image_canvas(image,
                                       output_path + '.skeleton.png',
                                       show=args.show,
                                       fig_width=args.figure_width,
                                       dpi_factor=args.dpi_factor) as ax:
                    skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)
                    a = plt.gcf()  
                    a.savefig('a.png')                
                    
            sys.exit(0)

if __name__ == '__main__':
    main()
