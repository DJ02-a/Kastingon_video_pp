# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

from scipy.signal import savgol_filter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from . import util
from .wholebody import Wholebody


def draw_pose_w_option(
    pose, H, W, draw_face=True, draw_hand=True, draw_body=True, wo_hand_kpts=True
):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if draw_hand:
        canvas = util.draw_handpose(canvas, hands, wo_kpts=wo_hand_kpts)
    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    return canvas


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


def simplify_facepose(faces):
    simple_faces = faces[[0], :4]

    # simple_faces[0, 0] = (faces[0,38] + faces[0,39]) /2 # left
    # simple_faces[0, 1] = (faces[0,41] + faces[0,42]) /2

    # simple_faces[0, 2] = (faces[0,44] + faces[0,45]) /2
    # simple_faces[0, 3] = (faces[0,47] + faces[0,48]) /2

    # only using mouth
    simple_faces[0, 0] = (faces[0, 49] + faces[0, 61]) / 2
    simple_faces[0, 1] = (faces[0, 55] + faces[0, 65]) / 2
    simple_faces[0, 2] = (faces[0, 52] + faces[0, 63]) / 2
    simple_faces[0, 3] = (faces[0, 58] + faces[0, 67]) / 2

    return simple_faces


class DWposeDetector:
    def __init__(self):
        pass

    def to(self, device):
        self.pose_estimation = Wholebody(device)
        return self

    def cal_height(self, input_image):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            # candidate[..., 0] /= float(W)
            # candidate[..., 1] /= float(H)
            body = candidate
        return body[0, ..., 1].min(), body[..., 1].max() - body[..., 1].min()

    #  #
    def get_batched_pose(
        self,
        frames,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        smooth=True,
        simple=True,
        savgol_window_len=15,
        **kwargs,
    ):
        candidate_list = []
        subset_list = []
        score_list = []
        input_image_list = []

        for input_image in frames:
            input_image = cv2.cvtColor(
                np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
            )

            input_image = HWC3(input_image)
            input_image = resize_image(input_image, detect_resolution)
            H, W, C = input_image.shape
            with torch.no_grad():
                candidate, subset = self.pose_estimation(input_image)

                candidate[..., 0] /= float(W)
                candidate[..., 1] /= float(H)

                score = subset[:, :18]
                max_ind = np.mean(score, axis=-1).argmax(axis=0)
                score = score[[max_ind]]

                candidate = candidate[[max_ind]]  # fix the 0-th dim of candidate as 1
                assert candidate.shape[0] == 1

                candidate_list.append(candidate)

                subset = subset[[max_ind]]

                subset_list.append(subset)

                for i in range(len(score)):
                    for j in range(len(score[i])):
                        if score[i][j] > 0.3:
                            score[i][j] = int(18 * i + j)
                        else:
                            score[i][j] = -1
                score_list.append(score)

                input_image_list.append(Image.fromarray(input_image))

        candidate_list = np.array(candidate_list).reshape(-1, 134, 2)

        subset_list = np.array(subset_list).reshape(-1, 134)

        cand1 = candidate_list[:, :, 0]
        cand2 = candidate_list[:, :, 1]

        if smooth:
            # cand1 = savgol_filter(candidate_list[:, :, 0], savgol_window_len, 3, axis=0)
            # cand2 = savgol_filter(candidate_list[:, :, 1], savgol_window_len, 3, axis=0)
            cand1 = gaussian_filter1d(candidate_list[:, :, 0], sigma=1.0, axis=0)
            cand2 = gaussian_filter1d(candidate_list[:, :, 1], sigma=1.0, axis=0)

        candidate_list_smoothed = np.stack([cand1, cand2], axis=-1)

        # subset_list_smoothed = gaussian_filter1d(subset_list, sigma=3.0, axis=0)

        nums, keys, locs = candidate.shape

        detected_map_list = []
        detected_map_sp_list = []
        for i in range(len(candidate_list_smoothed)):
            candidate = candidate_list_smoothed[[i], ...]

            un_visible = subset_list[[i], ...] < 0.3
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]
            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            body = candidate[:, :18].copy()
            body = body.reshape(18, -1)

            bodies = dict(candidate=body, subset=score_list[i])

            pose = dict(bodies=bodies, hands=hands, faces=faces)
            detected_map = draw_pose(pose, H, W)

            if simple:
                # faces_sp = simplify_facepose(faces)
                faces_sp = faces
                pose_sp = dict(bodies=bodies, hands=hands, faces=faces_sp)
                detected_map_sp = draw_pose_w_option(
                    pose_sp,
                    H,
                    W,
                    draw_hand=True,
                    draw_face=True,
                    wo_hand_kpts=True,
                )

            detected_map = HWC3(detected_map)
            detected_map_sp = HWC3(detected_map_sp)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(
                detected_map, (W, H), interpolation=cv2.INTER_LINEAR
            )

            detected_map_sp = cv2.resize(
                detected_map_sp, (W, H), interpolation=cv2.INTER_LINEAR
            )

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)
                detected_map_sp = Image.fromarray(detected_map_sp)

            detected_map_list.append(detected_map)
            detected_map_sp_list.append(detected_map_sp)
        return (
            detected_map_list,
            detected_map_sp_list,
            input_image_list,
        )

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        simple=True,
        **kwargs,
    ):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            # NOTE: hand detection
            # print(util.handDetect(candidate, subset, input_image))
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)

            score = subset[:, :18]
            max_ind = np.mean(score, axis=-1).argmax(axis=0)

            score = score[[max_ind]]
            body = candidate[:, :18].copy()
            body = body[[max_ind]]
            nums = 1
            body = body.reshape(nums * 18, locs)
            body_score = copy.deepcopy(score)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            # un_visible = subset < 0.3
            un_visible = subset < 0.5
            candidate[un_visible] = -1

            foot = candidate[:, 18:24]

            faces = candidate[[max_ind], 24:92]

            hands = candidate[[max_ind], 92:113]
            hands = np.vstack([hands, candidate[[max_ind], 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            detected_map = draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(
                detected_map, (W, H), interpolation=cv2.INTER_LINEAR
            )

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)

            if simple:
                # faces_sp = simplify_facepose(faces)
                faces_sp = faces
                pose_sp = dict(bodies=bodies, hands=hands, faces=faces_sp)
                detected_map_sp = draw_pose_w_option(
                    pose_sp,
                    H,
                    W,
                    draw_hand=True,
                    draw_face=True,
                    wo_hand_kpts=True,
                )

                detected_map_sp = cv2.resize(
                    detected_map_sp, (W, H), interpolation=cv2.INTER_LINEAR
                )

                if output_type == "pil":
                    detected_map_sp = Image.fromarray(detected_map_sp)

            else:
                detected_map_sp = None

            # input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_img = Image.fromarray(input_image)

            return (
                detected_map,
                detected_map_sp,
                input_img,
                body_score,
            )
