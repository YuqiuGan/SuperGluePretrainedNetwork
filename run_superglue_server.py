from __future__ import print_function

from pathlib import Path
import tqdm
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


class NetworkExtractor:
    def __init__(self, port):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://localhost:{self.port}")
        print("connected to server")

    def extract(self, img, frame_idx):
        # send the image as a multipart message
        msg0 = (
            np.array([img.shape[0], img.shape[1]])
            .reshape(-1)
            .astype(np.int32)
            .tobytes()
        )
        msg1 = img.astype(np.uint8).reshape(-1).tobytes()
        self.socket.send_multipart([msg0, msg1], 0)
        # receive the keypoints
        msgs = self.socket.recv_multipart(0)
        assert len(msgs) == 3, "#msgs={}".format(len(msgs))
        num_feat = np.frombuffer(msgs[0], dtype=np.int32)[0]
        feat_dim = np.frombuffer(msgs[0], dtype=np.int32)[1]
        xys = np.frombuffer(msgs[1], dtype=np.float32)
        if xys.shape[0] == num_feat * 2:
            xys = xys.reshape(num_feat, 2)
        elif xys.shape[0] == num_feat * 3:
            xys = xys.reshape(num_feat, 3)[:, :2]
        else:
            raise ValueError("xys.shape[0]={}".format(xys.shape[0]))
        desc = np.frombuffer(msgs[2], dtype=np.float32).reshape(num_feat, feat_dim)
        scores = np.ones(num_feat)
        return xys, desc, scores


class SuperGlueMatcher:
    def __init__(
        self,
        extractor,
        feature_type="super_point",
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
        # extractor
        self.extractor = extractor
        self.feature_type = feature_type
        # matcher config
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

    def match(self, image_query, image_train, threshold=0.7):
        # extract keypoints/descriptors for a single image
        xys_query, desc_query, scores_query = self.extractor.extract(image_query, 0)
        xys_train, desc_train, scores_train = self.extractor.extract(image_train, 1)
        # prepare super_glue matching
        if self.feature_type == "super_point":
            image_query_gray = cv2.cvtColor(image_query, cv2.COLOR_RGB2GRAY)
            image_train_gray = cv2.cvtColor(image_train, cv2.COLOR_RGB2GRAY)
            image_query_tensor = frame2tensor(image_query_gray, self.device)
            image_train_tensor = frame2tensor(image_train_gray, self.device)
            query_data = {
                "keypoints0": [torch.from_numpy(xys_query).float().to(self.device)],
                "descriptors0": [
                    torch.from_numpy(desc_query.T).float().to(self.device)
                ],
                "scores0": [torch.from_numpy(scores_query).float().to(self.device)],
                "image0": image_query_tensor,
            }
            train_data = {
                "keypoints1": [torch.from_numpy(xys_train).float().to(self.device)],
                "descriptors1": [
                    torch.from_numpy(desc_train.T).float().to(self.device)
                ],
                "scores1": [torch.from_numpy(scores_train).float().to(self.device)],
                "image1": image_train_tensor,
            }
            pred = self.matching({**query_data, **train_data})
            matches = pred["matches0"][0].cpu().numpy()
            confidence = pred["matching_scores0"][0].cpu().numpy()
        else:
            raise ValueError("Unknown feature_type={}".format(self.feature_type))

        if len(matches) == 0:
            return []

        # prepare visualization
        valid = matches > -1
        xys_query_valid = xys_query[valid]
        xys_train_valid = xys_train[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(xys_query), len(xys_train)),
            "Matches: {}".format(len(xys_query_valid)),
        ]

        # visualize matches
        img_vis = make_matching_plot_fast(
            image_query_gray,
            image_train_gray,
            xys_query,
            xys_train,
            xys_query_valid,
            xys_train_valid,
            color,
            text,
            path=None,
            show_keypoints=True,
        )

        cv2.imshow("SuperGlue", img_vis)
        cv2.waitKey(0)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SuperGlue demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--type", type=str, default="orb", help="feature type")
    parser.add_argument("--save_dir", type=str, default=None, help="save directory")
    parser.add_argument(
        "--query_dir", type=str, default=None, help="query image directory"
    )
    parser.add_argument(
        "--train_dir", type=str, default=None, help="train image directory"
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

    # prepare path
    if args.save_dir is not None:
        if os.path.exists(args.save_dir) is False:
            os.makedirs(args.save_dir)

    extractor = NetworkExtractor(args.port)
    matcher = SuperGlueMatcher(
        extractor,
        args.type,
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

    # Go through all images in the query and train directories
    for query_image_name in tqdm.tqdm(os.listdir(args.query_dir)):
        query_image = cv2.imread(os.path.join(args.query_dir, query_image_name))
        for train_image_name in tqdm.tqdm(os.listdir(args.train_dir)):
            train_image = cv2.imread(os.path.join(args.train_dir, train_image_name))
            matches = matcher.match(query_image, train_image)
