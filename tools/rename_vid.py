# I want to rename all the videos in a directory to a specific format

import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import glob
import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image

# video path 
video_dir = "/home/jgkwak/Moore-AnimateAnyone/cherry"
dwpose_video_dir = "/home/jgkwak/Moore-AnimateAnyone/cherry_dwpose"


# get all video list
video_list = glob.glob(os.path.join(video_dir, "*/*.mp4"))


new_video_dir = "/home/jgkwak/Moore-AnimateAnyone/cherry_dataset/cherry_all"
new_video_dwpose_dir = "/home/jgkwak/Moore-AnimateAnyone/cherry_dataset/cherry_all_dwpose"
# rename video and rename corresponding dwpose file
i = 0
for video_path in video_list:
    
    video_name = os.path.basename(video_path)
    video_type = os.path.basename(os.path.dirname(video_path))
    
    dwpath = os.path.join(dwpose_video_dir , video_type + "_dwpose", video_name)
    
    # new_video_name = video_name.replace(" ", "_")
    # reanme video with padded index
    new_video_name = f"{video_type}_{video_name}"
    new_video_path = os.path.join(new_video_dir, new_video_name)
    new_dwpose_path = os.path.join(new_video_dwpose_dir, new_video_name)
    os.rename(dwpath, new_dwpose_path)
    os.rename(video_path, new_video_path)
    
    print(f"Renamed {video_path} to {new_video_path}")
    i += 1 

print("Done")

