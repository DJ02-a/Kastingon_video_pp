import multiprocessing
import os

import cv2
import parmap


def crop_image(image_path):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    h, w, c = image.shape
    if w == 1024 and h == 1920:
        crop_amount = h - 1536
        top, bottom = crop_amount // 3 * 2, crop_amount // 3
    elif w == 512 and h == 960:
        crop_amount = h - 768
        top, bottom = crop_amount // 3 * 2, crop_amount // 3
    crop_image = image[top + bottom :, :, :]
    save_path = os.path.dirname(image_path).replace("output", "output_crop_top")
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, image_name), crop_image)

    crop_image = image[top:-bottom:, :, :]
    save_path = os.path.dirname(image_path).replace("output", "output_crop")
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, image_name), crop_image)


frame_root = "./assets/output_inference_frame_gaussian_1"

tdl: list[str] = []
for root, dirs, files in os.walk(frame_root):
    video_name = os.path.basename(root)
    for pp_data in dirs:
        if os.listdir(os.path.join(root, pp_data))[0].endswith(".jpg"):
            for image_name in os.listdir(os.path.join(root, pp_data)):
                tdl.append(os.path.join(frame_root, video_name, pp_data, image_name))

num_cores = multiprocessing.cpu_count()

parmap.map(crop_image, tdl, pm_pbar=True, pm_processes=int(num_cores * 0.75))
