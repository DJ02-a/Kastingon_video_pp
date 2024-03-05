import concurrent.futures
import os
import random
import sys
from pathlib import Path

from _src.dwpose import DWposeDetector
from _src.utils.util import get_fps, read_frames_cv, save_videos_from_pil


def process_single_video(
    video_path, detector, root_dir, save_dir, save_frames=True, smooth_pose=False
):
    relative_path = os.path.relpath(video_path, root_dir)
    out_path = os.path.join(save_dir, relative_path)
    out_path_sp = os.path.join(save_dir + "_simple", relative_path)
    if os.path.exists(out_path):
        return

    output_dir = Path(os.path.dirname(os.path.join(save_dir, relative_path)))
    output_dir_sp = Path(
        os.path.dirname(os.path.join(save_dir + "_simple", relative_path))
    )
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_sp.mkdir(parents=True, exist_ok=True)

    fps = get_fps(video_path)

    frames = read_frames_cv(video_path)

    print(
        "INFO:",
        relative_path,
        video_path,
        root_dir,
        "fps:",
        fps,
        "num_frames:",
        len(frames),
    )
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
    # add parent directory to save frames name: /path/to/video_dataset/*/*.mp4 -> /path/to/video_dataset_frames/imgs/*/*.jpg
    img_path = os.path.join(save_dir, "imgs", relative_path)
    frames = frames[::interval]
    if save_frames:
        os.makedirs(img_path.replace(".mp4", "_frames"), exist_ok=True)
        os.makedirs(img_path.replace(".mp4", "_dwpose"), exist_ok=True)
    frames = frames[:-1]

    if not smooth_pose:
        for i, frame_pil in enumerate(frames):
            # print(frame_pil.shape)
            result, _, result_sp, input_img = detector(frame_pil)
            # score = np.mean(score, axis=-1)

            kps_results.append(result)
            kps_results_sp.append(result_sp)

            input_img.save(
                os.path.join(img_path.replace(".mp4", "_frames"), f"{i:05d}.jpg")
            )
            result_sp.save(
                os.path.join(img_path.replace(".mp4", "_dwpose"), f"{i:05d}.jpg")
            )

    else:
        result, result_sp, input_img = detector.get_batched_pose(frames, smooth=True)
        for i, (r, r_sp, img) in enumerate(zip(result, result_sp, input_img)):
            kps_results.append(r)
            kps_results_sp.append(r_sp)
            img.save(os.path.join(img_path.replace(".mp4", "_frames"), f"{i:05d}.jpg"))
            r_sp.save(os.path.join(img_path.replace(".mp4", "_dwpose"), f"{i:05d}.jpg"))

    save_videos_from_pil(kps_results, out_path, fps=new_fps)
    save_videos_from_pil(kps_results_sp, out_path_sp, fps=new_fps)


def process_batch_videos(
    video_list, detector, root_dir, save_dir, simple, smooth=False
):
    # if simple:
    #     print("Simple mode")
    #     for i, video_path in enumerate(video_list):
    #         print(f"Process {i}/{len(video_list)} video")
    #         process_single_video_simple(video_path, detector, root_dir, save_dir)
    # else:
    print("Normal mode")
    for i, video_path in enumerate(video_list):
        print(f"Process {i}/{len(video_list)} video")
        process_single_video(
            video_path, detector, root_dir, save_dir, smooth_pose=smooth
        )


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
    parser.add_argument("--simple", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("-j", type=int, default=4, help="Num workers")

    args = parser.parse_args()
    num_workers = args.j
    if args.save_dir is None:
        save_dir = args.video_root + "_dwpose"
        if args.simple:
            save_dir += "_simple"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    else:
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
    batch_size = (len(video_mp4_paths) + num_workers - 1) // num_workers
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
            detector = DWposeDetector()
            # torch.cuda.set_device(gpu_id)
            detector = detector.to(f"cuda:{gpu_id}")

            futures.append(
                executor.submit(
                    process_batch_videos,
                    chunk,
                    detector,
                    args.video_root,
                    save_dir,
                    args.simple,
                    args.smooth,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()
