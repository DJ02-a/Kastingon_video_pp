import concurrent.futures
import glob
import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from moore_preprocess.dl_models.dwpose import DWposeDetector
from moore_preprocess.dl_models.VideoMatting import RobudstVideoMatting
from moore_preprocess.utils.util import (
    get_fps,
    get_grid_video,
    read_frames_cv,
    save_videos_from_pil,
    sharpening_filter,
)


def process_single_video(
    video_path,
    detector,
    matter,
    args,
):
    # set_directory
    folder_name = video_path.split("/")[-2]
    video_name = os.path.basename(video_path).split(".")[0]
    if folder_name != "video":
        workspace_folder_name = f"{folder_name}_{video_name}"
    else:
        workspace_folder_name = video_name
    workspace_dir = os.path.join(args.save_dir, workspace_folder_name)
    frame_path = os.path.join(workspace_dir, "frames")
    simple_openpose_path = os.path.join(workspace_dir, f"dwpose_simple")
    video_save_path = os.path.join(workspace_dir, "videos")
    grid_video_save_path = os.path.join(args.save_dir, "grid_videos")

    os.makedirs(frame_path, exist_ok=True)
    os.makedirs(simple_openpose_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)
    os.makedirs(grid_video_save_path, exist_ok=True)

    fps = get_fps(video_path)

    frames = read_frames_cv(video_path)

    print("INFO:", video_name, "fps:", fps, "num_frames:", len(frames))

    # if fps>40, then we can skip some frames
    if fps > 40:
        interval = 2
        new_fps = fps // 2
        print("changed fps:", fps, "->", new_fps)
    else:
        interval = 1
        new_fps = int(round(fps))

    frames = frames[::interval][:-1]
    kps_results_sp = []

    os.makedirs(frame_path, exist_ok=True)
    print("PROCESS : SAVE FRAMES")
    filter_frames = []
    for i, origin_frame in tqdm(enumerate(frames)):
        filter_frame = cv2.filter2D(origin_frame, -1, sharpening_filter)
        cv2.imwrite(os.path.join(frame_path, f"{i:05d}.jpg"), origin_frame)
        filter_frames.append(filter_frame)

    if args.matte_video:
        print("PROCESS : VIDEO MATTING")

        matte_path = os.path.join(workspace_dir, "matte")
        matte_frame_path = os.path.join(workspace_dir, "matte_frames")
        os.makedirs(matte_path, exist_ok=True)
        os.makedirs(matte_frame_path, exist_ok=True)
        matter(
            frame_path,
            output_type="png_sequence",
            output_alpha=matte_path,
            downsample_ratio=None,
            seq_chunk=12,
            num_workers=4,
        )
        matte_frames = []
        alpha_paths = sorted(glob.glob(matte_path + "/*.png"))
        for frame, alpha_path in zip(frames, alpha_paths):
            file_name = os.path.basename(alpha_path).split(".")[0]
            alpha = cv2.imread(alpha_path) / 255
            matte_frame = frame * alpha
            cv2.imwrite(
                os.path.join(matte_frame_path, str(file_name).zfill(5) + ".png"),
                matte_frame,
            )
            matte_frames.append(
                Image.fromarray(matte_frame[:, :, ::-1].astype(np.uint8))
            )
        save_videos_from_pil(
            matte_frames,
            os.path.join(video_save_path, "matte_video.mp4"),
            fps=new_fps,
        )

    print("PROCESS : DWPOSE")
    if not args.smooth:
        for i, frame in tqdm(enumerate(frames)):
            _, result_sp, _, _ = detector(
                frame,
                simple=args.simple,
            )

            kps_results_sp.append(result_sp)
            result_sp.save(os.path.join(simple_openpose_path, f"{i:05d}.jpg"))
    else:
        _, result_sp, _ = detector.get_batched_pose(
            frames,
            simple=args.simple,
            smooth=args.smooth,
            savgol_window_len=args.savgol_window_len,
        )
        for i, r_sp in enumerate(result_sp):
            kps_results_sp.append(r_sp)
            r_sp.save(os.path.join(simple_openpose_path, f"{i:05d}.jpg"))

    save_videos_from_pil(
        kps_results_sp,
        os.path.join(video_save_path, "openpose_wo_face_handkpt.mp4"),
        fps=new_fps,
    )

    shutil.copy(video_path, os.path.join(video_save_path, "origin.mp4"))
    get_grid_video(
        frames,
        kps_results_sp,
        new_fps,
        os.path.join(grid_video_save_path, f"{workspace_folder_name}.mp4"),
    )


def process_batch_videos(video_list, detector, videomatter, args):
    print("Normal mode")
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, videomatter, args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_root",
        type=str,
        default="./assets/video",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./assets/output_video",
        help="Path to save extracted pose videos",
    )
    parser.add_argument("--smooth", type=bool, default=True)
    parser.add_argument("--savgol_window_len", type=int, default=15)
    parser.add_argument("--simple", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--matte_video", type=bool, default=True)
    parser.add_argument("--remove_legacy", type=bool, default=False)
    parser.add_argument("--sp_wo_hand_kpts", type=bool, default=True)
    parser.add_argument("--sp_draw_hand", type=bool, default=True)
    parser.add_argument("--sp_draw_face", type=bool, default=True)
    parser.add_argument("--crop_result", type=bool, default=True)
    args = parser.parse_args()

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    print(f"avaliable gpu ids: {gpu_ids}")

    # collect all video_folder paths
    video_mp4_paths = set()
    for root, dirs, files in os.walk(args.video_root):
        for name in files:
            if name.endswith(".mp4"):
                video_mp4_paths.add(os.path.join(root, name))
    video_mp4_paths = list(video_mp4_paths)
    random.shuffle(video_mp4_paths)

    # split into chunks,
    batch_size = (len(video_mp4_paths) + args.num_workers - 1) // args.num_workers
    print(f"Num videos: {len(video_mp4_paths)} {batch_size = }")
    video_chunks = [
        video_mp4_paths[i : i + batch_size]
        for i in range(0, len(video_mp4_paths), batch_size)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(video_chunks):
            # init detector
            gpu_id = gpu_ids[i % len(gpu_ids)]

            videomatter = RobudstVideoMatting()
            detector = DWposeDetector()

            # torch.cuda.set_device(gpu_id)
            detector = detector.to(f"cuda:{gpu_id}")

            futures.append(
                executor.submit(
                    process_batch_videos, chunk, detector, videomatter, args
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
