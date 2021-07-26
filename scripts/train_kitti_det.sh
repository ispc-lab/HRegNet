python train_feats.py --batch_size 8 --epochs 100 --lr 0.001 --seed 1 --gpu GPU \
--npoints 16384 --dataset kitti --voxel_size 0.3 --ckpt_dir CKPT_DIR \
--use_fps --use_weights --data_list ./data/kitti_list --runname RUNNAME --augment 0.5 \
--root DATA_ROOT --wandb_dir WANDB_DIR --use_wandb