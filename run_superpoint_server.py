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
from models.utils import (
    AverageTimer,
    VideoStreamer,
    make_matching_plot_fast,
    frame2tensor,
)

torch.set_grad_enabled(False)


class SuperPointExtractor:
    def __init__(
        self,
        force_cpu=False,
        top_k=None,
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=-1,
        superglue="indoor",
        sinkhorn_iterations=20,
        match_threshold=0.2,
        save_results=False,
    ):

        self.device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        print('Running inference on device "{}"'.format(self.device))

        self.config = {
            "superpoint": {
                "nms_radius": nms_radius,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints,
            },
            "superglue": {
                "weights": superglue,
                "sinkhorn_iterations": sinkhorn_iterations,
                "match_threshold": match_threshold,
            },
        }
        self.matching = Matching(self.config).eval().to(self.device)
        self.save_results = save_results  # Add to config
        self.top_k = top_k

    def extract(self, frame):
        frame_tensor = frame2tensor(frame, self.device)
        last_data = self.matching.superpoint({"image": frame_tensor})

        xys = last_data["keypoints"][0].cpu().numpy()
        desc = (
            last_data["descriptors"][0].cpu().numpy().T
        )  # Transpose to be compatible with ORB
        scores = last_data["scores"][0].cpu().numpy()
        idxs = scores.argsort()[-self.top_k or None :]

        return xys[idxs], desc[idxs], scores[idxs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SuperGlue demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="0",
        help="ID of a USB webcam, URL of an IP camera, "
        "or path to an image directory or movie file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where to write output frames (If None, no output)",
    )
    parser.add_argument(
        "--image_glob",
        type=str,
        nargs="+",
        default=["*.png", "*.jpg", "*.jpeg"],
        help="Glob if a directory of images is specified",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Images to skip if input is a movie or directory",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000000,
        help="Maximum length if input is a movie or directory",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running inference. If two numbers, "
        "resize to the exact dimensions, if one number, resize the max "
        "dimension, if -1, do not resize",
    )
    parser.add_argument(
        "--superglue",
        choices={"indoor", "outdoor"},
        default="indoor",
        help="SuperGlue weights",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=-1,
        help="Maximum number of keypoints detected by Superpoint"
        " ('-1' keeps all keypoints)",
    )
    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.005,
        help="SuperPoint keypoint detector confidence threshold",
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=4,
        help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations performed by SuperGlue",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
    )
    parser.add_argument(
        "--show_keypoints", action="store_true", help="Show the detected keypoints"
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Do not display images to screen. Useful if running remotely",
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
    )
    parser.add_argument("--port", type=int, default=5555, help="port to map server")
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to file"
    )
    parser.add_argument(
        "--top-k", type=int, default=500, help="Number of top keypoints to keep"
    )
    args = parser.parse_args()
    print(args)

    if len(args.resize) == 2 and args.resize[1] == -1:
        args.resize = args.resize[0:1]
    if len(args.resize) == 2:
        print("Will resize to {}x{} (WxH)".format(args.resize[0], args.resize[1]))
    elif len(args.resize) == 1 and args.resize[0] > 0:
        print("Will resize max dimension to {}".format(args.resize[0]))
    elif len(args.resize) == 1:
        print("Will not resize images")
    else:
        raise ValueError("Cannot specify more than two integers for --resize")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = f"tcp://*:{args.port}"
    print("port", port)
    socket.bind(port)

    extractor = SuperPointExtractor(
        args.force_cpu,
        args.top_k,
        args.nms_radius,
        args.keypoint_threshold,
        args.max_keypoints,
        args.superglue,
        args.sinkhorn_iterations,
        args.match_threshold,
        args.save_results,
    )
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

        # preprocess image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # extract keypoints/descriptors
        xys, desc, scores = extractor.extract(image)
        num_feat = len(xys)
        feat_dim = desc.shape[1]
        print(f"num_feat={num_feat}, feat_dim={feat_dim}.")
        print(
            f"xys.shape={xys.shape}, desc.shape={desc.shape}, scores.shape={scores.shape}."
        )
        msg = np.array([num_feat, feat_dim]).reshape(-1).astype(np.int32).tobytes()
        socket.send(msg, 2)
        msg = xys.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 2)
        msg = desc.astype(np.float32).reshape(-1).tobytes()
        socket.send(msg, 0)
