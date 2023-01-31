from pathlib import Path
from tqdm import tqdm
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
import os

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

def write_xml(kp, des, score, FName):
    cv_file = cv2.FileStorage(FName, cv2.FILE_STORAGE_WRITE)
    cv_file.write("superpoint_keypoints", kp)
    cv_file.write("superpoint_descriptors", des)
    cv_file.write("superpoint_score", score)
    cv_file.release() 
    return True

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

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
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
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    # frame, ret = vs.next_frame()
    # assert ret, 'Error when reading the first frame (try different --input?)'

    # frame_tensor = frame2tensor(frame, device)
    
    collected_data = []
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished extracting superpoint intereting points!')
            break
        
        frame_tensor = frame2tensor(frame, device)
        # last_data is a dictionary
        # {
        #     'keypoints': keypoints,
        #     'scores': scores,
        #     'descriptors': descriptors,
        # }
        last_data = matching.superpoint({'image': frame_tensor})
        collected_data.append(last_data)

    # save superpoint results
    frame_list = list(Path(opt.input).glob(opt.image_glob[0]))
    frame_list.sort()
    
    for index, frame in enumerate(tqdm(frame_list)):
    # for index in range(0, len(frame_list)):
        # item = frame_list[index]
        frame_name = str(frame).split("/")[-1]
        file_name = frame_name.replace("masked", "superpoint").replace("png", "xml")
        file_name = os.path.join(opt.output_dir, file_name)
        # print(type(collected_data[index]['keypoints']))
        # print(collected_data[index]['keypoints'])
        # print(index)
        write_xml(np.asarray(collected_data[index]['keypoints'][0].cpu()), 
                  np.asarray(collected_data[index]['descriptors'][0].cpu()), 
                  np.asarray(collected_data[index]['scores'][0].cpu()),
                  file_name)
    print("Saved superpoint results to " + opt.output_dir)