import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from src.dwpose import DWposeDetector
from src.utils.util import get_fps, read_frames, save_videos_from_pil


def pose_smoothing(kps_results):
    return gaussian_filter1d(kps_results, sigma=1.0, axis=0)
    
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default='/home/jgkwak/dataset/TikTok_dataset/00088/images/')
    parser.add_argument("--smoothing_pose", type=bool, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        raise ValueError(f"Path: {args.video_path} not exists")

    dir_path, video_name = (
        os.path.dirname(args.video_path),
        os.path.splitext(os.path.basename(args.video_path))[0],
    )
    out_path = os.path.join(dir_path, video_name + "_kps.mp4")

    detector = DWposeDetector()
    detector = detector.to(f"cuda")
    
    f_names = glob.glob(os.path.join(args.video_path, '*.png'))
    
    print(len(f_names))
    fps = 8
    # kps_results = []
    frame_pil_list = []
    
    for i, frame in enumerate(f_names):
        
        frame_pil = Image.open(frame)
        frame_pil_list.append(frame_pil)
        
        
    kps_results, score = detector.get_batched_pose(frame_pil_list)
    # score = np.mean(score, axis=-1)

    # kps_results.append(result)

            
    print(out_path)
    save_videos_from_pil(kps_results, out_path, fps=fps)

    # fps = get_fps(args.video_path)
    # frames = read_frames(args.video_path)
    
    # kps_results = []
    # for i, frame_pil in enumerate(frames):
    #     result, score = detector(frame_pil)
    #     score = np.mean(score, axis=-1)

    #     kps_results.append(result)

    # if args.smoothing_pose:
    #     kps_results = pose_smoothing(kps_results)
        
    # print(out_path)
    # save_videos_from_pil(kps_results, out_path, fps=fps)
