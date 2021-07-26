import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

import os
import glob
import numpy as np
import MinkowskiEngine as ME

from models.utils import generate_rand_rotm, generate_rand_trans, apply_transform

def read_kitti_bin_voxel(filename, npoints=None, voxel_size=None):
    '''
    Input:
        filename
        npoints: int/None
        voxel_size: int/None
    '''
    scan = np.fromfile(filename, dtype=np.float32, count=-1).reshape([-1,4])
    scan = scan[:,:3]

    if voxel_size is not None:
        _, sel = ME.utils.sparse_quantize(scan / voxel_size, return_index=True)
        scan = scan[sel]
    if npoints is None:
        return scan.astype('float32')
    
    N = scan.shape[0]
    if N >= npoints:
        sample_idx = np.random.choice(N, npoints, replace=False)
    else:
        sample_idx = np.concatenate((np.arange(N), np.random.choice(N, npoints-N, replace=True)), axis=-1)
    
    scan = scan[sample_idx, :].astype('float32')
    return scan

class KittiDataset(Dataset):
    '''
    Params:
        root
        seqs
        npoints
        voxel_size
        data_list
        augment
    '''
    def __init__(self, root, seqs, npoints, voxel_size, data_list, augment=0.0):
        super(KittiDataset, self).__init__()

        self.root = root
        self.seqs = seqs
        self.npoints = npoints
        self.voxel_size = voxel_size
        self.augment = augment
        self.data_list = data_list
        self.dataset = self.make_dataset()
    
    def make_dataset(self):
        last_row = np.zeros((1,4), dtype=np.float32)
        last_row[:,3] = 1.0
        dataset = []

        for seq in self.seqs:
            fn_pair_poses = os.path.join(self.data_list, seq + '.txt')

            with open(fn_pair_poses, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data_dict = {}
                    line = line.strip(' \n').split(' ')
                    src_fn = os.path.join(self.root, seq, 'velodyne', line[0] + '.bin')
                    dst_fn = os.path.join(self.root, seq, 'velodyne', line[1] + '.bin')
                    values = []
                    for i in range(2, len(line)):
                        values.append(float(line[i]))
                    values = np.array(values).astype(np.float32)
                    rela_pose = values.reshape(3,4)
                    rela_pose = np.concatenate([rela_pose, last_row], axis = 0)
                    data_dict['points1'] = src_fn
                    data_dict['points2'] = dst_fn
                    data_dict['Tr'] = rela_pose
                    dataset.append(data_dict)
        
        return dataset
    
    def __getitem__(self, index):
        data_dict = self.dataset[index]
        src_points = read_kitti_bin_voxel(data_dict['points1'], self.npoints, self.voxel_size)
        dst_points = read_kitti_bin_voxel(data_dict['points2'], self.npoints, self.voxel_size)
        Tr = data_dict['Tr']
        
        # Random rotation augmentation (Only for training feature extraction)
        if np.random.rand() < self.augment:
            aug_T = np.zeros((4,4), dtype=np.float32)
            aug_T[3,3] = 1.0
            rand_rotm = generate_rand_rotm(1.0, 1.0, 45.0)
            aug_T[:3,:3] = rand_rotm
            src_points = apply_transform(src_points, aug_T)
            Tr = Tr.dot(np.linalg.inv(aug_T))
        
        src_points = torch.from_numpy(src_points)
        dst_points = torch.from_numpy(dst_points)
        Tr = torch.from_numpy(Tr)
        R = Tr[:3,:3]
        t = Tr[:3,3]
        return src_points, dst_points, R, t
    
    def __len__(self):
        return len(self.dataset)