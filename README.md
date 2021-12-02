# IDEAL-NeRF

### Environment

- Create an environment by
  - ```Plaintext
    conda env create -f environment.yml
    conda activate pytorch3d
    ```

- Face Tracker Set Up
  - Download Base Face Model From: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details 
  - Put "01_MorphableModel.mat" to data_util/face_tracking/3DMM/
  - ```Plaintext
    cd data_util/face_tracking
    
    python convert_BFM.py
    ```

### Data Process

> Raw Data From: https://1drv.ms/u/s!ArdWM1-cwOGGjC62-OQwRD9Kuj1b?e=SdlpjF

```Plaintext
python data_util/process_data.py --id=May
```

### Run Command

#### Config

```Bash
# 保存模型参数的位置
basedir=dataset/May/logs 
expname=deepspeech
# 数据源
datadir=dataset/May
gt_dirs=ori_imgs
aud_file=aud_may.npy
# tensorboard 可视化文件夹
vis_path=dataset/May/running/deepspeech
# 实验参数
N_sample=64
N_importance=128
num_work=1
batch_size=1
lrate=3e-4
N_iters=60
near=0.5772005200386048
far=1.1772005200386046
testskip=104
N_rand=3072
lc_weight=0.005
mouth_rays=512
torso_rays=0
dim_expr=79
dim_aud=29
```

#### Train

```Plaintext
CUDA_VISIBLE_DEVICES=7 nohup python -u NeRFs/HeadNeRF/train/audio_exp_nerf.py --config NeRFs/HeadNeRF/configs/audio_expr_nerf/may/ablation/deepspeech_audio > output/deepspeech.out &
```

#### Eval

> The hyperparameters can also be written in a config file.

```Plaintext
CUDA_VISIBLE_DEVICES=0 nohup python -u NeRFs/HeadNeRF/test/eval_aud_exp_nerf.py --basedir=dataset/Obama/logs --datadir=dataset/Obama --expname=blend_highlight_torso --evalExpr_path=dataset/May/transforms_exp_train.json --save_path=output/cross_subject_blend/Obama0_May_Expr --aud_file=dataset/audio/aud_may.npy --num_work=1 --batch_size=1 --testskip=1 --near=0.5674083709716797 --far=1.1674083709716796 > output/render_V_Obama0_A_May.out &
```
