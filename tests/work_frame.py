import concurrent.futures
import glob
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from moore_preprocess.dl_models.dwpose import DWposeDetector
from moore_preprocess.dl_models.VideoMatting import RobudstVideoMatting
from moore_preprocess.utils.util import get_grid_video, sharpening_filter


def process_single_video(
    video_info,
    detector,
    matter,
    args,
):
    # set_directory
    video_name, clip_num = video_info
    workspace_folder_name = f"{video_name}_{str(clip_num).zfill(3)}"
    workspace_dir = os.path.join(args.save_dir, workspace_folder_name)
    frame_path = os.path.join(workspace_dir, "frames")
    simple_openpose_path = os.path.join(workspace_dir, "dwpose_simple")

    os.makedirs(frame_path, exist_ok=True)
    os.makedirs(simple_openpose_path, exist_ok=True)

    print("INFO:", video_name)

    frames = []
    kps_results = []
    kps_results_sp = []

    origin_frame_path = os.path.join(args.frame_root, video_name, clip_num)
    for i in sorted(os.listdir(origin_frame_path)):
        frames.append(cv2.imread(os.path.join(origin_frame_path, i)))
    origin_frame_path = origin_frame_path[:-1]

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
        alpha_paths = sorted(glob.glob(matte_path + "/*.png"))
        for frame, alpha_path in zip(frames, alpha_paths):
            file_name = os.path.basename(alpha_path).split(".")[0]
            alpha = cv2.imread(alpha_path) / 255
            matte_frame = frame * alpha
            cv2.imwrite(
                os.path.join(matte_frame_path, str(file_name).zfill(5) + ".jpg"),
                matte_frame,
            )

    print("PROCESS : DWPOSE")
    if not args.smooth:
        for i, frame in tqdm(enumerate(filter_frames)):
            _, result_sp, _, _ = detector(
                frame,
                simple=args.simple,
            )

            kps_results_sp.append(result_sp)
            result_sp.save(os.path.join(simple_openpose_path, f"{i:05d}.jpg"))
    else:
        _, result_sp, _ = detector.get_batched_pose(
            filter_frames,
            simple=args.simple,
            smooth=args.smooth,
            savgol_window_len=args.savgol_window_len,
        )
        for i, r_sp in enumerate(result_sp):
            kps_results_sp.append(r_sp)
            r_sp.save(os.path.join(simple_openpose_path, f"{i:05d}.jpg"))

    grid_video_save_path = os.path.join(args.save_dir, "grid_videos")
    os.makedirs(grid_video_save_path, exist_ok=True)
    get_grid_video(
        frames,
        kps_results,
        fps=24,
        save_path=os.path.join(grid_video_save_path, f"{workspace_folder_name}.mp4"),
    )


def process_batch_videos(chunk_list, detector, videomatter, args):
    print("Normal mode")
    for i, frame_path in enumerate(chunk_list):
        print(f"Process {i}/{len(chunk_list)} video")
        process_single_video(frame_path, detector, videomatter, args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_root",
        type=str,
        default="./assets/inference_frame",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./assets/output_inference_frame_gaussian_1",
        help="Path to save extracted pose videos",
    )
    parser.add_argument("--smooth", default=True)
    parser.add_argument("--savgol_window_len", type=int, default=15)
    parser.add_argument("--simple", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--matte_video", default=True)
    args = parser.parse_args()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    print(f"avaliable gpu ids: {gpu_ids}")

    video_count = 0
    tdl = []
    for root, dirs, files in os.walk(args.frame_root):
        video_name = os.path.basename(root)
        for clip_num in dirs:
            if os.listdir(os.path.join(root, clip_num))[0].endswith(".jpg"):
                tdl.append([video_name, clip_num])
                video_count += 1

    # split into chunks,
    batch_size = (video_count + args.num_workers - 1) // args.num_workers
    print(f"Num videos: {video_count} {batch_size = }")
    frame_chunks = [tdl[i : i + batch_size] for i in range(0, video_count, batch_size)]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i, chunk in enumerate(frame_chunks):
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
