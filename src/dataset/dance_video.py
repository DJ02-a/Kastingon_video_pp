import json
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import os
import glob
import cv2

class HumanDanceVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fashion_meta.json"],
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        #TODO: change this back to kps_path!! (original)
        kps_path = video_meta["kps_path"]
        # kps_path = video_meta["simple_kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
            img = video_reader[index]
            vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = kps_reader[index]
            pose_pil_image_list.append(Image.fromarray(img.asnumpy()))

        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())

        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        return sample

    def __len__(self):
        return len(self.vid_meta)




class HumanDanceVideoDataset_fast(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        # data_meta_paths=["./data/fashion_meta.json"],
        data_path = "./data/youtube_8"
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        # vid_meta = []
        # for data_meta_path in data_meta_paths:
        #     vid_meta.extend(json.load(open(data_meta_path, "r")))
        # self.vid_meta = vid_meta


        video_names = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
        # video_names = [name for name in os.listdir(data_path) if os.path.isdir(name)]
        
        self.data = []
        for video_name in video_names:

            if 'frames' in video_name:
                
                # print('Reading video... :', video_name)
                # Divide each video into 2 videos
                
                frame_list = glob.glob(os.path.join(data_path, video_name,"*.jpg"))
                dwpose_list = glob.glob(os.path.join(data_path, video_name.replace('_frames', '_dwpose'),"*.jpg"))
                #matte_list = glob.glob(os.path.join(data_path, video_name.replace('_frames', '_composition_matte'),"*.jpg"))
                
                num_frames = len(frame_list)
                
                frame_list.sort()
                dwpose_list.sort()
                #matte_list.sort()
                
                dwposes1 = [Image.open(dwpose_path).convert("RGB") for dwpose_path in dwpose_list[:num_frames//2]]
                frames1 = [Image.open(frame_path).convert("RGB") for frame_path in frame_list[:num_frames//2]] 
                # frames1 = [cv2.imread(frame_path) for frame_path in frame_list[:num_frames//2]]
                #matte_list1 = [cv2.imread(matte_path) for matte_path in matte_list[:num_frames//2]]
                assert len(frames1) == len(dwposes1), f"{len(frames1) = } != {len(dwposes1) = }"
                
                dwposes2 = [Image.open(dwpose_path).convert("RGB") for dwpose_path in dwpose_list[num_frames//2:]]
                frames2 = [Image.open(frame_path).convert("RGB") for frame_path in frame_list[num_frames//2:]] 
                # frames2 = [cv2.imread(frame_path) for frame_path in frame_list[num_frames//2:]]
                #matte_list2 = [cv2.imread(matte_path) for matte_path in matte_list[num_frames//2:]]
                
                assert len(frames2) == len(dwposes2), f"{len(frames2) = } != {len(dwposes2) = }"
            
                self.data.append(dict(frames=frames1, dwposes=dwposes1))
                self.data.append(dict(frames=frames2, dwposes=dwposes2))


        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        # video_meta = self.vid_meta[index]
        # video_path = video_meta["video_path"]
        # #TODO: change this back to kps_path!! (original)
        # kps_path = video_meta["kps_path"]
        # # kps_path = video_meta["simple_kps_path"]

        # video_reader = VideoReader(video_path)
        # kps_reader = VideoReader(kps_path)
        

        video_reader = self.data[index]['frames']
        kps_reader = self.data[index]['dwposes']
        #matte_reader = self.data[index]['mattes']

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = }"

        video_length = len(video_reader)

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        vid_pil_image_list = []
        pose_pil_image_list = []
        for index in batch_index:
             
            
            # img = video_reader[index]
            # vid_pil_image_list.append(Image.fromarray(img.asnumpy()))
            
            
            img_pil = video_reader[index]
            #mat = matte_reader[index]
            
            # img_mat = img / 255. * mat / 255. 
            # img_mat = (img_mat * 255).astype(np.uint8)
            # img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)
            # img_pil = Image.fromarray(img_mat.astype('uint8'))
            
            vid_pil_image_list.append(img_pil)
            
            pose_pil_image = kps_reader[index]
            pose_pil_image_list.append(pose_pil_image)

        # ref_img_idx = random.randint(0, video_length - 1)
        # ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())
        
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img = video_reader[ref_img_idx]
        # ref_matte = matte_reader[ref_img_idx]
        # ref_img_mat = ref_img / 255. * ref_matte / 255.
        # ref_img_mat = (ref_img_mat * 255).astype(np.uint8)
        # ref_img_mat = cv2.cvtColor(ref_img_mat, cv2.COLOR_BGR2RGB)
        # ref_img = Image.fromarray(ref_img_mat)
        
        # transform
        state = torch.get_rng_state()
        pixel_values_vid = self.augmentation(
            vid_pil_image_list, self.pixel_transform, state
        )
        pixel_values_pose = self.augmentation(
            pose_pil_image_list, self.cond_transform, state
        )
        pixel_values_ref_img = self.augmentation(ref_img, self.pixel_transform, state)
        clip_ref_img = self.clip_image_processor(
            images=ref_img, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            #video_dir=video_path,
            pixel_values_vid=pixel_values_vid,
            pixel_values_pose=pixel_values_pose,
            pixel_values_ref_img=pixel_values_ref_img,
            clip_ref_img=clip_ref_img,
        )

        return sample

    def __len__(self):
        # return len(self.vid_meta)
        return len(self.data)