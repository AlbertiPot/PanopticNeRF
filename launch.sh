python train_net.py \
--cfg_file ./PanopticNeRF/configs/panopticnerf_test.yaml \
pretrain nerf \
gpus '0,' \
use_stereo True \
use_pspnet True \
use_depth True \
pseudo_filter True \
weight_th 0.05 \
resume False