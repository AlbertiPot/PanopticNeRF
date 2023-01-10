# # # train
# python train_net.py \
# --cfg_file ./configs/panopticnerf_test.yaml \
# gpus [1] \
# pretrain nerf \
# use_stereo True \
# use_pspnet True \
# use_depth True \
# pseudo_filter True \
# weight_th 0.05 \
# resume False

# Render semantic map, panoptic map and depth map in a single forward pass
python run.py \
--type visualize \
--cfg_file configs/panopticnerf_test.yaml \
use_stereo False