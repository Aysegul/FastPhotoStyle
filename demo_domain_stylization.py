"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
import numpy as np
import random
import os
import torch
import process_stylization_domain
from photo_wct import PhotoWCT

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--content_image_path',default='/datasets/RAND_CITYSCAPES/resized/RGB/',help='path to train')
parser.add_argument('--content_seg_path', default=[])
parser.add_argument('--style_image_path',default='/datasets/cityscapes_raw/resized/',help='path to train')
parser.add_argument('--style_seg_path', default=[])
parser.add_argument('--content_list', default='datasets/lists/synthia_image_list.txt', help='List')
parser.add_argument('--style_list', default='datasets/lists/cityscapes_train_image_list.txt', help='List')
parser.add_argument('--num_outputs', type=int, default=10, help='how many stylized versions')
parser.add_argument('--out_path', default='./results/output', help='folder to output images')
parser.add_argument('--fast', action='store_true', default=True)
parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
args = parser.parse_args()

# Load model
p_wct = PhotoWCT(min_mask=10)
p_wct.load_state_dict(torch.load(args.model))

# Create output folder
try:
    os.makedirs(args.out_path)
except OSError:
    pass

if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=45, eps=0.3)
else:
    from photo_smooth import Propagator
    p_pro = Propagator()
if args.cuda:
    p_wct.cuda(0)

# Read content image list
with open(args.content_list) as f:
    content_list = f.readlines()
content_list = [x.strip() for x in content_list]

# Read style image list
with open(args.style_list) as f:
    style_list = f.readlines()
style_list = [x.strip() for x in style_list]

content_list.sort()
np.random.shuffle(style_list)

count = 0
n_style_img = len(style_list)
for ix, c_f in enumerate(content_list):
    content_image_path = os.path.join(args.content_image_path, c_f)
    if type(args.content_seg_path) is list:
        content_seg_path =  args.content_seg_path
    else:
        content_seg_path = os.path.join(args.content_seg_path, c_f)

    k=0
    for j in range(0,50*args.num_outputs):
        s_f = style_list[count]
        count += 1
        if count >= n_style_img:
            count = 0
        style_image_path = os.path.join(args.style_image_path, s_f)
        if type(args.style_seg_path) is list:
            style_seg_path = args.style_seg_path
        else:
            style_seg_path = os.path.join(args.style_seg_path, s_f)

        o_f = c_f[0:-4] + '_' + s_f.replace('/','_')
        output_image_path = os.path.join(args.out_path, o_f)
        print(output_image_path)
        try:
            os.makedirs(os.path.dirname(output_image_path))
        except OSError:
            pass

        try:
            process_stylization_domain.domain_stylization(
                stylization_module=p_wct,
                smoothing_module=p_pro,
                content_image_path=content_image_path,
                style_image_path=style_image_path,
                content_seg_path=content_seg_path,
                style_seg_path=style_seg_path,
                output_image_path=output_image_path,
                cuda=args.cuda,
            )
            k = k+1
        except:
            pass
        if k>= args.num_outputs:
            break
