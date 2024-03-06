# Moore-AnimateAnyone_INVZ 

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

#### 변경점 (vs. Moore)

Moore에서는 facial landmark를 모두 입력으로 사용하였지만, 현재 제외한 상태입니다.
따라서 body와 hand에 관한 pose만 사용중입니다.   

원래는 RGB 비디오를 입력으로 하여 코드를 실행할 경우 (rgb-video-root) rgb-video-root_dwpose 라는 폴더내에 같은 이름의 skeleton 비디오를 저장하지만, 각 프레임과 그에 해당하는 스켈레톤 이미지들을 저장하도록 바꿨습니다. + 비디오도 같이 저장합니다.

dwpose에서 unvisible의 score threshold를 0.3 -> 0.5 로 변경했습니다. 

손가락마디들의 line만 사용하고 keypoint를 제거하였습니다.

기존의 stage1.yaml 과 stage2.yaml에서  use_8bit_adam=False로 설정되어있지만
아무리해도 finetune1.yaml 및 finetune2.yaml에서는  OOM이 나서 False로 수정해주고 나니 해결했습니다.


## 1. Preprocessing 

- 비디오 입력으로 부터 각 프레임마다 Openpose 결과를 뽑아내는 과정입니다. 
- [DWpose](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet) 모델을 사용합니다.
- training 과 inference에서 모두 preprocessing이 필요합니다.
- Skeleton pose 정보가 아닌 이미지 자체를 condition으로 사용합니다.

#### 실행 

```shell
python tests/test.py
```


#### Hyper-parameter 

Preprocessing에서 바꿀수 있는 하이퍼파라미터들은 다음과 같습니다. (현재는 ./src/dwpose/__init__.py 에서 하드코딩하고 있고 추후 변경할 예정입니다.)

- simple: face나 hand landmark를 제거하기 위해 둔 파라미터이고 True시 결과 비디오를 _dw_pose_sp 폴더에 따로 저장합니다. -> 지금은 hand keypoint를 제외한 landmark를 모두 사용하고 있습니다.
- simple의 경우 draw_pose_w_option 함수를 통해 원하는 landmark를 제거할 수 있게 했습니다. 

