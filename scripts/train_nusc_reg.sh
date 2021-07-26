python train_reg.py --batch_size 16 --epochs 100 --lr 0.001 --seed 1 --gpu 1 \
--npoints 8192 --dataset nuscenes --voxel_size 0.3 --ckpt_dir CKPT_DIR \
--use_fps --use_weights --alpha 2.0 \
--data_list ./data/nuscenes_list --runname RUNNAME --augment 0.0 \
--root DATA_ROOT --wandb_dir WANDB_DIR --freeze_detector \
--pretrain_feats PRETRAIN_FEATS --use_wandb