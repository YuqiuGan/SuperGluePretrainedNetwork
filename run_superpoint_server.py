from __future__ import print_function

from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import os
import zmq
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

class SuperPointExtractor():

    def __init__(self, opt):

        self.device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        self.vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)

        self.config = {
            'superpoint': {
                'nms_radius': opt.nms_radius,
                'keypoint_threshold': opt.keypoint_threshold,
                'max_keypoints': opt.max_keypoints
            },
            'superglue': {
                'weights': opt.superglue,
                'sinkhorn_iterations': opt.sinkhorn_iterations,
                'match_threshold': opt.match_threshold,
            }
        }
        self.matching = Matching(self.config).eval().to(self.device)
        self.colleted_data = []
        self.save_results = opt.save_results # Add to config
        

    def extract(self, frame):
        frame_tensor = frame2tensor(frame, self.device)
        last_data = self.matching.superpoint({'image': frame_tensor})
        if self.save_results:
            self.colleted_data.append(last_data)
        return last_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
        '--port', type=int, default=5555,
        help="port to map server"
    )

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')
    
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = f"tcp://*:{opt.port}"
    print("port", port)
    socket.bind(port)

    extractor = SuperPointExtractor(opt)
    while True:
        print(f"SuperPoint listending to {port}")
        msgs = socket.recv_multipart(0)
        assert len(msgs) == 2, "#msgs={}".format(len(msgs))
        wh = np.frombuffer(msgs[0], dtype=np.int32)
        W = wh[0]
        H = wh[1]
        print(f"W={W}, H={H}")
        msg = msgs[1]
        image = np.frombuffer(msg, dtype=np.uint8).reshape(H, W, -1).squeeze()

        # # preprocess image
        # image = norm_RGB(image)[None] 
        # if iscuda: image = image.cuda()

        # extract keypoints/descriptors
        sp_results = extractor.extract(image)
        kp = sp_results['keypoints'][0].cpu()
        desc = sp_results['descriptors'][0].cpu()
        scores = sp_results['scores'][0].cpu()

        num_feat = len(kp)
        feat_dim = desc.shape[1]
        print(f"num_feat={num_feat}, feat_dim={feat_dim}.")
        print(f"xys.shape={kp.shape}, desc.shape={desc.shape}, scores.shape={scores.shape}.")
        msg = np.array([num_feat, feat_dim]).reshape(-1).astype(np.int32).tobytes()
        socket.send(msg, 2)
        msg = kp.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 2)
        msg = desc.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 0)