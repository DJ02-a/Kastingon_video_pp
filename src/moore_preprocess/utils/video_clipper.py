import os
import shutil

import yaml


class Video_Clipper:
    def __init__(self, dataset_path, save_path, video_data_path):
        self.dataset_path = dataset_path
        self.video_names = os.listdir(self.dataset_path)
        self.save_path = save_path
        self.video_data_path = video_data_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.copy_folders = ["frames", "dwpose_simple"]
        return

    def cut(
        self,
        video_name,
        cut_sections,
    ):
        frame_num = len(
            os.listdir(os.path.join(self.dataset_path, video_name, "frames"))
        )
        alive_sections = self._get_alive_sections(frame_num, cut_sections)
        self._make_new_dataset(video_name, alive_sections)

        return alive_sections

    def create_video_info(self):
        data = {}
        for video_name in self.video_names:
            data[video_name] = [[25, 50], [50, 100]]

        with open("video_data.yaml", "w") as file:
            yaml.dump(data, file)

    def get_video_info(self):
        with open(self.video_data_path, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data

    def _get_alive_sections(self, frame_num, cut_sections):
        alive_sections = []

        for i, (cur_from, cur_idx) in enumerate(cut_sections):
            if i == 0:
                alive_section = [0, cur_from - 1]
            elif i + 1 == len(cut_sections):
                alive_section = [before_to + 1, frame_num - 1]
            else:
                alive_section = [before_to + 1, cur_from - 1]

            if alive_section[1] - alive_section[0] + 1 >= 24:
                alive_sections.append(alive_section)
            before_from, before_to = cur_from, cur_idx

        return alive_sections

    def _make_new_dataset(self, video_name, alive_sections):
        copy_path = os.path.join(self.dataset_path, video_name)
        for i, (from_idx, to_idx) in enumerate(alive_sections):
            for idx in range(from_idx, to_idx):
                for folder_name in self.copy_folders:
                    file_name = f"{idx:05d}.jpg"

                    new_save_path = os.path.join(
                        self.save_path, f"{video_name}_{i:02d}", folder_name
                    )
                    if not os.path.exists(new_save_path):
                        os.makedirs(new_save_path)
                    shutil.copy(
                        os.path.join(copy_path, folder_name, file_name),
                        os.path.join(new_save_path, f"{file_name}"),
                    )
        return


if __name__ == "__main__":
    vc = Video_Clipper(
        dataset_path="./assets/output",
        save_path="./assets/output_new",
        video_data_path="./assets/video_data.yaml",
    )
    vc.create_video_info()
    videos_data = vc.get_video_info()
    for vdieo_name, cut_sections in videos_data.items():
        vc.cut(vdieo_name, cut_sections)
