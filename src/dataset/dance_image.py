import json
import random

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import os
import glob
import cv2
import numpy as np 


class HumanDanceDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
        use_simple_kps=True,
        max_frames = -1
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        self.vid_meta = vid_meta

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
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
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio
        self.use_simple_kps = use_simple_kps
        self.max_frames = max_frames

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]
        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        
        if self.use_simple_kps:
            simple_kps_path = video_meta["simple_kps_path"]
            simple_kps_reader = VideoReader(simple_kps_path)

        
        # if self.max_frames > 0:
        #     video_reader = video_reader[:self.max_frames]
        #     kps_reader = kps_reader[:self.max_frames]   
        #     simple_kps_reader = simple_kps_reader[:self.max_frames]

        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = } in {video_path}"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        ref_img = video_reader[ref_img_idx]
        ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img = video_reader[tgt_img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose = kps_reader[tgt_img_idx]
        tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())
        
        if self.use_simple_kps:
            simple_tgt_pose = simple_kps_reader[tgt_img_idx]
            simple_tgt_pose_pil = Image.fromarray(simple_tgt_pose.asnumpy())

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        
        # simple_tgt_pose_img = self.augmentation(simple_tgt_pose_pil, self.cond_transform, state)
        
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )
        return sample

    def __len__(self):
        return len(self.vid_meta)





class HumanDanceDataset_fast_matte(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        #data_meta_paths=["./data/fahsion_meta.json"],
        data_path = "./data/youtube_8",
        sample_margin=30,
        use_simple_kps=False,
        max_frames = -1
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
      
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
                matte_list = glob.glob(os.path.join(data_path, video_name.replace('_frames', '_composition_matte'),"*.jpg"))
                
                num_frames = len(frame_list)
                
                frame_list.sort()
                dwpose_list.sort()
                matte_list.sort()
                
                dwposes1 = [Image.open(dwpose_path).convert("RGB") for dwpose_path in dwpose_list[:num_frames//2]]
                # frames1 = [Image.open(frame_path).convert("RGB") for frame_path in frame_list[:num_frames//2]] 
                frames1 = [cv2.imread(frame_path) for frame_path in frame_list[:num_frames//2]]
                matte_list1 = [cv2.imread(matte_path) for matte_path in matte_list[:num_frames//2]]
                assert len(frames1) == len(dwposes1), f"{len(frames1) = } != {len(dwposes1) = }"
                
                dwposes2 = [Image.open(dwpose_path).convert("RGB") for dwpose_path in dwpose_list[num_frames//2:]]
                frames2 = [cv2.imread(frame_path) for frame_path in frame_list[num_frames//2:]]
                matte_list2 = [cv2.imread(matte_path) for matte_path in matte_list[num_frames//2:]]
                
                assert len(frames2) == len(dwposes2), f"{len(frames2) = } != {len(dwposes2) = }"
            
                self.data.append(dict(frames=frames1, dwposes=dwposes1, mattes = matte_list1))
                self.data.append(dict(frames=frames2, dwposes=dwposes2, mattes = matte_list2))
                
            
            

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
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
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio
        self.use_simple_kps = use_simple_kps
        self.max_frames = max_frames

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):

        #NOTE: we dont need to read each video for every iteration, we can read all the videos at init stage

        # video_meta = self.vid_meta[index]
        # video_path = video_meta["video_path"]
        # kps_path = video_meta["kps_path"]
        # video_reader = VideoReader(video_path)
        # kps_reader = VideoReader(kps_path)
        
        # if self.use_simple_kps:
        #     simple_kps_path = video_meta["simple_kps_path"]
        #     simple_kps_reader = VideoReader(simple_kps_path)

        
        # if self.max_frames > 0:
        #     video_reader = video_reader[:self.max_frames]
        #     kps_reader = kps_reader[:self.max_frames]   
        #     simple_kps_reader = simple_kps_reader[:self.max_frames]

        video_reader = self.data[index]['frames']
        kps_reader = self.data[index]['dwposes']
        matte_reader = self.data[index]['mattes']
        
        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = }"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)


        ref_img = video_reader[ref_img_idx]
        # ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)   
        ref_matte = matte_reader[ref_img_idx]
        
        ref_img_matte =  ref_img / 255. * ref_matte / 255.
        ref_img_matte = (ref_img_matte * 255).astype(np.uint8)
        ref_img_matte = cv2.cvtColor(ref_img_matte, cv2.COLOR_BGR2RGB)
        ref_img_pil = Image.fromarray((ref_img_matte).astype('uint8'))
        
        # save the ref_img_pil
        ref_img_pil.save('/home/jgkwak/Moore-AnimateAnyone/Moore-AnimateAnyone/for_save/ref_img_pil.jpg')
        
        
        
        tgt_img = video_reader[tgt_img_idx]
        # tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)   
        tgt_matte = matte_reader[tgt_img_idx]

        tgt_img_matte =  tgt_img / 255. * tgt_matte / 255.
        tgt_img_matte = (tgt_img_matte * 255).astype(np.uint8)
        tgt_img_matte = cv2.cvtColor(tgt_img_matte, cv2.COLOR_BGR2RGB)
        tgt_img_pil = Image.fromarray((tgt_img_matte).astype('uint8'))
        
        # ref_img_pil = Image.fromarray(ref_img)
        # tgt_img_pil = video_reader[tgt_img_idx]

        # ref_img_pil = video_reader[ref_img_idx]
        # ref_img = video_reader[ref_img_idx]
        # ref_img_pil = Image.fromarray(ref_img.asnumpy())
        
        # tgt_img = video_reader[tgt_img_idx]
        # tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose_pil = kps_reader[tgt_img_idx]
        # tgt_pose = kps_reader[tgt_img_idx]
        # tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())
        
        # if self.use_simple_kps:
        #     simple_tgt_pose = simple_kps_reader[tgt_img_idx]
        #     simple_tgt_pose_pil = Image.fromarray(simple_tgt_pose.asnumpy())

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        
        # simple_tgt_pose_img = self.augmentation(simple_tgt_pose_pil, self.cond_transform, state)
        
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
           # video_dir=video_path,
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )
        return sample

    def __len__(self):
        return len(self.data)
    
    



class HumanDanceDataset_fast(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        #data_meta_paths=["./data/fahsion_meta.json"],
        data_path = "./data/youtube_8",
        sample_margin=30,
        use_simple_kps=False,
        max_frames = -1
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.sample_margin = sample_margin

        # -----
        # vid_meta format:
        # [{'video_path': , 'kps_path': , 'other':},
        #  {'video_path': , 'kps_path': , 'other':}]
        # -----
      
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
                # matte_list1 = [cv2.imread(matte_path) for matte_path in matte_list[:num_frames//2]]
                assert len(frames1) == len(dwposes1), f"{len(frames1) = } != {len(dwposes1) = }"
                
                dwposes2 = [Image.open(dwpose_path).convert("RGB") for dwpose_path in dwpose_list[num_frames//2:]]
                frames2 = [Image.open(frame_path).convert("RGB") for frame_path in frame_list[num_frames//2:]]
                # frames2 = [cv2.imread(frame_path) for frame_path in frame_list[num_frames//2:]]
                # matte_list2 = [cv2.imread(matte_path) for matte_path in matte_list[num_frames//2:]]
                
                assert len(frames2) == len(dwposes2), f"{len(frames2) = } != {len(dwposes2) = }"
            
                self.data.append(dict(frames=frames1, dwposes=dwposes1))
                self.data.append(dict(frames=frames2, dwposes=dwposes2))


        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
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
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio
        self.use_simple_kps = use_simple_kps
        self.max_frames = max_frames

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):

        #NOTE: we dont need to read each video for every iteration, we can read all the videos at init stage

        # video_meta = self.vid_meta[index]
        # video_path = video_meta["video_path"]
        # kps_path = video_meta["kps_path"]
        # video_reader = VideoReader(video_path)
        # kps_reader = VideoReader(kps_path)
        
        # if self.use_simple_kps:
        #     simple_kps_path = video_meta["simple_kps_path"]
        #     simple_kps_reader = VideoReader(simple_kps_path)

        
        # if self.max_frames > 0:
        #     video_reader = video_reader[:self.max_frames]
        #     kps_reader = kps_reader[:self.max_frames]   
        #     simple_kps_reader = simple_kps_reader[:self.max_frames]

        video_reader = self.data[index]['frames']
        kps_reader = self.data[index]['dwposes']
        #matte_reader = self.data[index]['mattes']
        
        assert len(video_reader) == len(
            kps_reader
        ), f"{len(video_reader) = } != {len(kps_reader) = }"

        video_length = len(video_reader)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)


        ref_img_pil = video_reader[ref_img_idx]
        
        # ref_img_pil = Image.fromarray(ref_img.asnumpy())
        tgt_img_pil = video_reader[tgt_img_idx]
        
        # tgt_img_pil = Image.fromarray(tgt_img.asnumpy())

        tgt_pose_pil = kps_reader[tgt_img_idx]
        
        # tgt_pose_pil = Image.fromarray(tgt_pose.asnumpy())
        

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_pose_img = self.augmentation(tgt_pose_pil, self.cond_transform, state)
        
        # simple_tgt_pose_img = self.augmentation(simple_tgt_pose_pil, self.cond_transform, state)
        
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=ref_img_pil, return_tensors="pt"
        ).pixel_values[0]

        sample = dict(
            img=tgt_img,
            tgt_pose=tgt_pose_img,
            ref_img=ref_img_vae,
            clip_images=clip_image,
        )
        return sample

    def __len__(self):
        return len(self.data)