import concurrent.futures
import glob
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from moore_preprocess.dl_models.dwpose import DWposeDetector
from moore_preprocess.dl_models.VideoMatting import RobudstVideoMatting
from moore_preprocess.utils.util import sharpening_filter


def process_single_video(
    video_info,
    detector,
    matter,
    args,
):
    # set_directory
    video_name, clip_num = video_info
    workspace_folder_name = (
        f"{video_name}_{str(clip_num).zfill(3)}_sm{str(args.smooth)[0]}"
    )
    workspace_dir = os.path.join(args.save_dir, workspace_folder_name)
    frame_path = os.path.join(workspace_dir, "frames")
    sharp_frame_path = os.path.join(workspace_dir, "sharp_frames")
    openpose_path = os.path.join(workspace_dir, "dwpose")
    simple_name = "simple"
    simple_openpose_path = os.path.join(workspace_dir, f"dwpose_{simple_name}")
    simple_openpose_woface_path = os.path.join(
        workspace_dir, f"dwpose_woface_{simple_name}"
    )
    os.makedirs(frame_path, exist_ok=True)
    os.makedirs(sharp_frame_path, exist_ok=True)
    os.makedirs(openpose_path, exist_ok=True)
    os.makedirs(simple_openpose_path, exist_ok=True)
    os.makedirs(simple_openpose_woface_path, exist_ok=True)

    print("INFO:", video_name)

    kps_results = []
    kps_results_sp = []
    kps_results_sp_woface = []

    frames = []
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
        cv2.imwrite(os.path.join(sharp_frame_path, f"{i:05d}.jpg"), filter_frame)
        filter_frames.append(filter_frame)

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

    print("PROCESS : DWPOSE")
    if not args.smooth:
        for i, frame in tqdm(enumerate(frames)):
            result, result_sp, result_sp_woface, _, _ = detector(
                frame,
                simple=args.simple,
                sp_draw_hand=args.sp_draw_hand,
                sp_draw_face=args.sp_draw_face,
                sp_wo_hand_kpts=args.sp_wo_hand_kpts,
            )

            kps_results.append(result)
            kps_results_sp.append(result_sp)
            kps_results_sp_woface.append(result_sp_woface)
            result.save(os.path.join(openpose_path, f"{i:05d}.jpg"))
            result_sp.save(os.path.join(simple_openpose_path, f"{i:05d}.jpg"))
            result_sp_woface.save(
                os.path.join(simple_openpose_woface_path, f"{i:05d}.jpg")
            )
    else:
        result, result_sp, result_sp_woface, _ = detector.get_batched_pose(
            frames,
            simple=args.simple,
            smooth=args.smooth,
            sp_draw_hand=args.sp_draw_hand,
            sp_draw_face=args.sp_draw_face,
            sp_wo_hand_kpts=args.sp_wo_hand_kpts,
        )
        for i, (r, r_sp, r_sp_woface) in enumerate(
            zip(result, result_sp, result_sp_woface)
        ):
            kps_results.append(r)
            kps_results_sp.append(r_sp)
            kps_results_sp_woface.append(r_sp_woface)
            r.save(os.path.join(openpose_path, f"{i:05d}.jpg"))
            r_sp.save(os.path.join(simple_openpose_path, f"{i:05d}.jpg"))
            r_sp_woface.save(os.path.join(simple_openpose_woface_path, f"{i:05d}.jpg"))


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
        default="./assets/frame",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./assets/output_frame",
        help="Path to save extracted pose videos",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./assets/dataset",
    )
    parser.add_argument("--smooth", default=True)
    parser.add_argument("--simple", default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--matte_video", default=True)
    parser.add_argument("--remove_legacy", default=False)

    parser.add_argument("--sp_wo_hand_kpts", default=True)
    parser.add_argument("--sp_draw_hand", default=True)
    parser.add_argument("--sp_draw_face", default=True)
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
