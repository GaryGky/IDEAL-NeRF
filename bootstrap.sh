# 训练Stage2
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u NeRFs/HeadNeRF/train/head_baseline.py --config=NeRFs/HeadNeRF/configs/head_baseline.txt > output/head_baseline.out &

# 测试视频
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u NeRFs/HeadNeRF/test/test_nerf.py --num_work=1 --batch_size=1 --testskip=1 --perturb=0 --N_importance=128 --N_sample=64 --datadir=dataset/Obama_base --basedir=dataset/Obama/logs --expname=base_parallel --aud_file=aud_ch.npy --near=0.7047980308532715 --far=1.3047980308532714 &

# space GPU
CUDA_VISIBLE_DEVICES=0 nohup python data_util/face_parsing/resnet.py && sleep 1d &

# Face Tracker
CUDA_VISIBLE_DEVICES=7 nohup python -u  data_util/face_tracking/face_tracker.py --idname=May --img_h=450 --img_w=450 --frame_num=8000 > Tracker_Obama.out &

# self attention
CUDA_VISIBLE_DEVICES=4 nohup python -u NeRFs/HeadNeRF/train/attention_nerf.py --config=NeRFs/HeadNeRF/configs/attention_nerf/obama3/attention_highlight_lc_big.txt > output/obama_attention_nerf.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u NeRFs/HeadNeRF/train/attention_nerf.py --config=NeRFs/HeadNeRF/configs/attention_nerf/obama3/attention_all_inputs.txt > output/attention_all_inputs.out &

# 不分离head和Torso 用Aud_Expr_NeRF进行训练
CUDA_VISIBLE_DEVICES=0 nohup python -u NeRFs/HeadNeRF/train/audio_exp_nerf.py --config=NeRFs/HeadNeRF/configs/audio_expr_nerf/obama/paper_model.txt > output/render_obama_torso.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u NeRFs/HeadNeRF/train/audio_exp_nerf.py --config=NeRFs/HeadNeRF/configs/audio_expr_nerf/obama3/blend_highlight.txt > output/obama3_blend.out &
CUDA_VISIBLE_DEVICES=2 nohup python -u NeRFs/HeadNeRF/train/audio_exp_nerf.py --config=NeRFs/HeadNeRF/configs/audio_expr_nerf/may/blend_highlight.txt > output/may_blend.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u NeRFs/HeadNeRF/train/audio_exp_nerf.py --config=NeRFs/HeadNeRF/configs/audio_expr_nerf/Chinese/blend_highlight.txt > output/chn_blend.out &

# test Obama3
CUDA_VISIBLE_DEVICES=0 nohup python -u NeRFs/HeadNeRF/test/eval_aud_exp_nerf.py --basedir=dataset/Obama3/logs --datadir=dataset/Obama3 --expname=blend_highlight_torso --evalExpr_path=dataset/Chinese/transforms_exp_train.json --save_path=output/cross_subject_blend/Obama_Chn_Expr --aud_file=dataset/audio/aud_ch.npy --num_work=1 --batch_size=1 --testskip=1 --near=0.6338776230812073 --far=1.2338776230812072 > output/render_V_Obama_A_Chn.out &

# test Chinese
CUDA_VISIBLE_DEVICES=0 nohup python -u NeRFs/HeadNeRF/test/eval_aud_exp_nerf.py --basedir=dataset/Chinese/logs --datadir=dataset/Chinese --expname=blend_highlight_torso --evalExpr_path=dataset/May/transforms_exp_train.json --save_path=output/cross_subject_blend/Chn_May_Expr --aud_file=dataset/audio/aud_may.npy --num_work=1 --batch_size=1 --testskip=1 --near=0.6020160794258118 --far=1.2020160794258117 > output/render_V_Chn_A_May.out &

# test Obama
CUDA_VISIBLE_DEVICES=0 nohup python -u NeRFs/HeadNeRF/test/eval_aud_exp_nerf.py --basedir=dataset/Obama/logs --datadir=dataset/Obama --expname=blend_highlight_torso --evalExpr_path=dataset/May/transforms_exp_train.json --save_path=output/cross_subject_blend/Obama0_May_Expr --aud_file=dataset/audio/aud_may.npy --num_work=1 --batch_size=1 --testskip=1 --near=0.5674083709716797 --far=1.1674083709716796 > output/render_V_Obama0_A_May.out &


CUDA_VISIBLE_DEVICES=7 nohup python -u NeRFs/HeadNeRF/train/audio_exp_nerf.py --config NeRFs/HeadNeRF/configs/audio_expr_nerf/may/ablation/deepspeech_audio > output/deepspeech.out &
