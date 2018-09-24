"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import time
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch
from smooth_filter import smooth_filter
from process_stylization import Timer, memory_limit_image_resize



def domain_stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path, style_seg_path, output_image_path, cuda):
    # Load image
    with torch.no_grad():
        cont_img = Image.open(content_image_path).convert('RGB')
        styl_img = Image.open(style_image_path).convert('RGB')

        new_cw, new_ch = memory_limit_image_resize(cont_img)
        new_sw, new_sh = memory_limit_image_resize(styl_img)
        cont_pilimg = cont_img.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height
        try:
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)
            cont_seg.resize((new_cw,new_ch),Image.NEAREST)
            styl_seg.resize((new_sw,new_sh),Image.NEAREST)

        except:
            cont_seg = []
            styl_seg = []

        cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
        styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)

        if cuda:
            cont_img = cont_img.cuda(0)
            styl_img = styl_img.cuda(0)
            stylization_module.cuda(0)

        # cont_img = Variable(cont_img, volatile=True)
        # styl_img = Variable(styl_img, volatile=True)

        cont_seg = np.asarray(cont_seg)
        styl_seg = np.asarray(styl_seg)

        with Timer("Elapsed time in stylization: %f"):
            stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
        if ch != new_ch or cw != new_cw:
            print("De-resize image: (%d,%d)->(%d,%d)" %(new_cw,new_ch,cw,ch))
            stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')

        # save intermediate
        utils.save_image(stylized_img.data.cpu().float(), output_image_path, nrow=1, padding=0)

        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        with Timer("Elapsed time in propagation: %f"):
            out_img = smoothing_module.process(out_img, cont_pilimg)

        # sometimes opencv smoothing generates images with white regions - check for that.
        ndarr = np.array(out_img)
        num_total_pixels = ndarr.shape[0] * ndarr.shape[1]
        num_bad_pixels1 = np.sum(np.float32(
          np.logical_and(np.logical_and(ndarr[:, :, 0] == 255, ndarr[:, :, 1] == 255), ndarr[:, :, 2] == 255)))
        num_bad_pixels2 = np.sum(np.float32(
          np.logical_and(np.logical_and(ndarr[:, :, 0] == 0, ndarr[:, :, 1] == 0), ndarr[:, :, 2] == 0)))
        if num_bad_pixels1/num_total_pixels > 0.2 or num_bad_pixels2/num_total_pixels > 0.2:
            print('Not smoothed due to an opencv bug')
            return

        if not cuda:
            print("NotImplemented: The CPU version of smooth filter has not been implemented currently.")
            return

        out_img.save(output_image_path)

