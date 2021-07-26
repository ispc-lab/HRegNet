python train_reg.py --batch_size 16 --epochs 100 --lr 0.001 --seed 1 --gpu GPU \
--npoints 16384 --dataset kitti --voxel_size 0.3 --ckpt_dir CKPT_DIR \
--use_fps --use_weights --alpha 1.8 \
--data_list ./data/kitti_list --runname RUNNAME --augment 0.0 \
--root DATA_ROOT --wandb_dir WANDB_DIR --freeze_detector \
--pretrain_feats PRETRAIN_FEATS --use_wandb