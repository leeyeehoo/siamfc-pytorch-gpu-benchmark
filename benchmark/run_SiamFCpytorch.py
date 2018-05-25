from __future__ import division
import sys
CODE_ROOT = 'ROOT OF YOUR LOCAL FOLDER'
sys.path.insert(0, CODE_ROOT)
import h5py
import torch
import os
import numpy as np
from PIL import Image
# import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.siamese import SiameseNet
import collections
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
NET_PATH = 'TO YOUR WEIGHTS'
def load_net(fname, net):
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[k]))   
        v.copy_(param)
def convert_bbox_format(bbox, to = 'center-based'):
    x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
    if to == 'top-left-based':
        x -= get_center(target_width)
        y -= get_center(target_height)
    elif to == 'center-based':
        y += get_center(target_height)
        x += get_center(target_width)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(to))
    return Rectangle(x*1.0, y*1.0, target_width*1.0, target_height*1.0)
def get_center(x):
    return (x - 1.) / 2.

def run_SiamFCpytorch(seq, rp, bSaveImage):
    hp, evaluation, run, env, design = parse_arguments()
    #final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    final_score_sz = 265 
    siam = SiameseNet(env.root_pretrained, design.net)
    load_net(NET_PATH,siam)
    siam.cuda()
    
    frame_name_list = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format   
    
    init_bb = Rectangle(x - 1, y - 1, float(width), float(height))
    init_bb = convert_bbox_format(init_bb,'center-based')    
    
    bboxes, speed = tracker(hp, run, design, frame_name_list, init_bb.x, init_bb.y, init_bb.width, init_bb.height, final_score_sz,
                                siam, evaluation.start_frame)
    
    trajectory = [Rectangle(val[0] + 1, val[1] + 1, val[2], val[3]) for val in
                  bboxes]
    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = speed
    return result