## HRegNet: Efficient Hierarchical Point Cloud Registration Network

### Introduction
The repository contains the source code and pre-trained models of our paper (published on ICCV 2021): `HRegNet: Efficient Hierarchical Point Cloud Registration Network`.

### Environments
The code mainly requires the following libraries and you can check `requirements.txt` for more environment requirements.
- PyTorch 1.7.0/1.7.1
- Cuda 11.0/11.1
- [pytorch3d 0.3.0](https://github.com/facebookresearch/pytorch3d)
- [MinkowskiEngine 0.5](https://github.com/NVIDIA/MinkowskiEngine)

Please run the following commands to install `point_utils`
```
cd models/PointUtils
python setup.py install
```

**Training device**: NVIDIA RTX 3090

### Usages
The data should be organized as follows:
#### KITTI odometry dataset
```
DATA_ROOT
├── 00
│   ├── velodyne
│   ├── calib.txt
├── 01
├── ...
```
#### NuScenes dataset
```
DATA_ROOT
├── v1.0-trainval
│   ├── maps
│   ├── samples
│   │   ├──LIDAR_TOP
│   ├── sweeps
│   ├── v1.0-trainval
├── v1.0-test
│   ├── maps
│   ├── samples
│   │   ├──LIDAR_TOP
│   ├── sweeps
│   ├── v1.0-test
```
### Train
The training of the whole network is divided into two steps: we firstly train the feature extraction module and then train the network based on the pretrain features.
#### Train feature extraction
- Train keypoints detector by running `sh scripts/train_kitti_det.sh` or `sh scripts/train_nusc_det.sh`, please reminder to specify the `GPU`,`DATA_ROOT`,`CKPT_DIR`,`RUNNAME`,`WANDB_DIR` in the scripts.
- Train descriptor by running `sh scripts/train_kitti_desc.sh` or `sh scripts/train_nusc_desc.sh`, please reminder to specify the `GPU`,`DATA_ROOT`,`CKPT_DIR`,`RUNNAME`,`WANDB_DIR` and `PRETRAIN_DETECTOR` in the scripts.

#### Train the whole network
Train the network by running `sh scripts/train_kitti_reg.sh` or `sh scripts/train_nusc_reg.sh`, please reminder to specify the `GPU`,`DATA_ROOT`,`CKPT_DIR`,`RUNNAME`,`WANDB_DIR` and `PRETRAIN_FEATS` in the scripts.

### Test
We provide pretrain models in `ckpt/pretrained`, please run `sh scripts/test_kitti.sh` or `sh scripts/test_nusc.sh`, please reminder to specify `GPU`,`DATA_ROOT`,`SAVE_DIR` in the scripts. The test results will be saved in `SAVE_DIR`.