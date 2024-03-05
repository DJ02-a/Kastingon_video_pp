# Moore-AnimateAnyone_INVZ 

Moore AnimateAnyonem을 변형한 코드입니다.

- 프로세스는 크게 3가지로 나눌수 있습니다.
    - Preprocessing 
    - Inference
    - Finetuning


## 0. Prepare 

**Envs**
```shell
Conda create -n aa python=3.11

```

**Weight downloading**: 

아래 커맨드로 pretrained_weights 폴더에 pretrained model들을 준비합니다

```shell
python -m tools.download_weights

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt

conda activate aa
```
- 파일 구조 (./pretrained_weights/)
```text
./pretrained_weights/
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
|-- sd-vae-ft-mse
|   |-- config.json
|   |-- diffusion_pytorch_model.bin
|   `-- diffusion_pytorch_model.safetensors
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `--
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
python -m tools.extract_dwpose_from_vid --video_root rgb-video-root
```


#### Hyper-parameter 

Preprocessing에서 바꿀수 있는 하이퍼파라미터들은 다음과 같습니다. (현재는 ./src/dwpose/__init__.py 에서 하드코딩하고 있고 추후 변경할 예정입니다.)

- simple: face나 hand landmark를 제거하기 위해 둔 파라미터이고 True시 결과 비디오를 _dw_pose_sp 폴더에 따로 저장합니다. -> 지금은 hand keypoint를 제외한 landmark를 모두 사용하고 있습니다.
- simple의 경우 draw_pose_w_option 함수를 통해 원하는 landmark를 제거할 수 있게 했습니다. 


## 2. Inference

Inference는 기존 moore와 같게 openpose video입력과 reference image를 받아서 수행합니다. 

여기서 inference option은 animation.yaml 에서 지정할수 있습니다. 

animation.yaml내에서 model이나 inference image, driving video를 변경하시면됩니다.
model부분은 moore original을 사용하시면 그대로 두고 finetuned model을 사용하시면 바꿔야합니다. 

#### 실행 

```shell
python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 -L 384 --exp oompa
```

W: width H: height L: frame개수

## 3. Finetuning 

Finetuning은 크게 2step으로 나누어집니다. 

    Step1: image단위 training 
    Step2: video단위 training

### Stage1
    !Trainable: pose-guider, denoising_unet, reference_unet 원래의 moore train1.py는 network initialization을 SD로 하지만, finetune1.py에서는 pretrained moore modele들로 init 해주었습니다. 

    

#### 실행
```shell
accelerate launch finetune_stage_1.py --config configs/train/finetune_stage1.yaml 
```    
    주요 하이퍼파라미터 
    - data_path: dwpose 이미지들의 폴더 경로 
    - max_train_step 지정: finetuning 비디오가 적다면 작게 설정, 많다면 크게 설정 (500~5000)
    - exp_name, output_dir: ckpt들이 저장되는 경로 

#### 결과
- output_dir/exp_name 내에 pose-guider, reference_unet, denoising_unet이 저장


### Stage2 
    !Trainable: motion_module: mm은 stage2에서 새롭게 추가(animatediff 구조와같음) 나머지 freeze 원래의 moore train2.py는 network initialization을 animatediff의 mm으로 해주지만, finetune2.py에선 moore에서 finetuned된 mm을 initialization으로 사용하였습니다.  

#### 실행 
```shell
accelerate launch finetune_stage_2.py --config configs/train/finetune_stage2.yaml 
```  

    주요 하이퍼파라미터 
    - data_path: dwpose 이미지들의 폴더 경로 
    - max_train_step 지정: finetuning 비디오가 적다면 작게 설정, 많다면 크게 설정 (500~5000)
    - exp_name, output_dir: ckpt들이 저장되는 경로 

#### 결과 
- motion_module checkpoint 파일



