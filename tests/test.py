import concurrent.futures
import glob
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from moore_preprocess.dl_models.dwpose import DWposeDetector
from moore_preprocess.dl_models.VideoMatting import RobudstVideoMatting
from moore_preprocess.utils.util import get_fps, read_frames_cv, save_videos_from_pil


def process_single_video(
    video_path,
    detector,
    matter,
    args,
    save_dir,
):
    video_name = os.path.basename(video_path).split(".")[0]
    workspace_dir = os.path.join(save_dir, video_name)
    frame_path = os.path.join(workspace_dir, "frames")
    openpose_path = os.path.join(workspace_dir, "dwpose")
    simple_name = "simple"
    simple_openpose_path = os.path.join(workspace_dir, f"dwpose_{simple_name}")
    video_save_path = os.path.join(workspace_dir, "videos")
    folder_name = video_path.split("/")[-2]
    if folder_name != "video":
        dataset_folder_name = f"{folder_name}/{video_name}"
    else:
        dataset_folder_name = f"{video_name}"
    os.makedirs(frame_path, exist_ok=True)
    os.makedirs(openpose_path, exist_ok=True)
    os.makedirs(simple_openpose_path, exist_ok=True)
    os.makedirs(video_save_path, exist_ok=True)

    if os.path.exists(
        os.path.join(args.dataset_dir, dataset_folder_name, "origin.mp4")
    ):
        return

    fps = get_fps(video_path)

    frames = read_frames_cv(video_path)

    print("INFO:", video_name, "fps:", fps, "num_frames:", len(frames))

    kps_results = []
    kps_results_sp = []
    # if fps>40, then we can skip some frames
    if fps > 40:
        interval = 2
        new_fps = fps // 2
        print("changed fps:", fps, "->", new_fps)
    else:
        interval = 1
        new_fps = int(round(fps))

    frames = frames[::interval][:-1]
    os.makedirs(frame_path, exist_ok=True)
    print("PROCESS : SAVE FRAMES")
    for i, origin_frame in tqdm(enumerate(frames)):
        cv2.imwrite(os.path.join(frame_path, f"{i:05d}.jpg"), origin_frame)

    if args.matte_video:
        print("PROCESS : VIDEO MATTING")

        matte_frame_path = os.path.join(workspace_dir, "matte_frames")
        os.makedirs(matte_frame_path, exist_ok=True)
        matte_path = os.path.join(workspace_dir, "matte")
        os.makedirs(matte_path, exist_ok=True)

        matter(
            frame_path,
            output_type="png_sequence",
            output_alpha=matte_path,
            downsample_ratio=None,
            seq_chunk=12,
            num_workers=4,
        )
        matte_frames = []
        alpha_paths = sorted(glob.glob(matte_path + "/*.jpg"))
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
        for i, frame in tqdm(enumerate(matte_frames)):
            result, result_sp, _, _ = detector(
                frame,
                simple=args.simple,
                sp_draw_hand=args.sp_draw_hand,
                sp_draw_face=args.sp_draw_face,
                sp_wo_hand_kpts=args.sp_wo_hand_kpts,
            )

            kps_results.append(result)
            kps_results_sp.append(result_sp)

            result.save(os.path.join(openpose_path, f"{i:05d}.jpg"))
            result_sp.save(os.path.join(simple_openpose_path, f"{i:05d}.jpg"))

    else:
        result, result_sp, _ = detector.get_batched_pose(
            matte_frames,
            simple=args.simple,
            smooth=args.smooth,
            sp_draw_hand=args.sp_draw_hand,
            sp_draw_face=args.sp_draw_face,
            sp_wo_hand_kpts=args.sp_wo_hand_kpts,
        )
        for i, (r, r_sp) in enumerate(zip(result, result_sp)):
            kps_results.append(r)
            kps_results_sp.append(r_sp)
            r.save(os.path.join(openpose_path, f"{i:05d}.jpg"))
            r_sp.save(os.path.join(simple_openpose_path), f"{i:05d}.jpg")

    save_videos_from_pil(
        kps_results,
        os.path.join(video_save_path, "openpose_full.mp4"),
        fps=new_fps,
    )
    save_videos_from_pil(
        kps_results_sp,
        os.path.join(video_save_path, "openpose_simple.mp4"),
        fps=new_fps,
    )
    shutil.copy(video_path, os.path.join(video_save_path, "origin.mp4"))
    shutil.copytree(
        video_save_path,
        os.path.join(args.dataset_dir, f"{dataset_folder_name}"),
        dirs_exist_ok=True,
    )
    if args.remove_legacy:
        shutil.rmtree(workspace_dir)


def process_batch_videos(video_list, detector, videomatter, args, root_dir):
    print("Normal mode")
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(video_path, detector, videomatter, args, root_dir)


if __name__ == "__main__":
    # -----
    # NOTE:
    # python tools/extract_dwpose_from_vid.py --video_root /path/to/video_dir
    # -----
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
        default="./assets/output",
        help="Path to save extracted pose videos",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./assets/dataset",
    )
    parser.add_argument("--smooth", default=True)
    parser.add_argument("--simple", default=True)
    parser.add_argument("--num_workers", type=int, default=2, help="Num workers")
    parser.add_argument("--matte_video", default=True)
    parser.add_argument("--remove_legacy", default=True)

    parser.add_argument("--sp_wo_hand_kpts", default=True)
    parser.add_argument("--sp_draw_hand", default=True)
    parser.add_argument("--sp_draw_face", default=True)
    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
                    process_batch_videos,
                    chunk,
                    detector,
                    videomatter,
                    args,
                    save_dir,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
