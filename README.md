# Moore-AnimateAnyone_INVZ 

- pyproject.toml은 cuda 11.x 버전을 기준으로 작성되었습니다.

- cuda 12.x 버전일 경우 [링크](https://onnxruntime.ai/docs/install/)를 따를 것
- 밑의 onnxruntime-gpu가 이미 설치되어 있다면 삭제 후 reinstall
```shell
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

- [onnx-runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirementsto) cuda 버전 별 호환되는 onnxruntime 버전 확인

## 0. Prepare 

**Envs**
```shell
Conda create -n aa python=3.11
```

**Weight downloading**: 

아래 커맨드로 pretrained_weights 폴더에 pretrained model들을 준비합니다

```shell
# create conda env

pip install -e .

# download weights
python -m moore_preprocess.utils.download_weights
```

## Preprocessing 

- 비디오 입력으로 부터 각 프레임마다 Openpose 결과를 뽑아내는 과정입니다. 
- [DWpose](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet) 모델을 사용합니다.
- training 과 inference에서 모두 preprocessing이 필요합니다.
- Skeleton pose 정보가 아닌 이미지 자체를 condition으로 사용합니다.

#### 실행 

```shell
python tests/test.py
```

## Video Clipper
- './assets/video_data.yaml'에 있는 동영상 정보를 가지고 전처리된 데이터를 구역별로 잘라준다.
- video_data.yaml에는 자를 구간을 적어놔야 함


```python
from moore_preprocess.utils.video_clipper import Video_Clipper

# 경로 미리 결정
vc = Video_Clipper(
    dataset_path="./assets/output",
    save_path="./assets/output_new",
    video_data_path="./assets/video_data.yaml",
)

# 전처리가 한번 완료된 비디오 폴더들이 있는 곳을 지정. yaml파일에 이름으로 List를 만듬
vc.create_video_info()
# assets/output/grid_videos에서 자를 구간의 프레임을 yaml파일에 기록

# yaml 보고 자르기 시작
videos_data = vc.get_video_info()
for vdieo_name, cut_sections in videos_data.items():
    vc.cut(vdieo_name, cut_sections)
```