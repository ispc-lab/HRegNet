import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import KeypointDetector, DescExtractor, CoarseReg, FineReg, WeightedSVDHead

class HierFeatureExtraction(nn.Module):
    def __init__(self, args):
        super(HierFeatureExtraction, self).__init__()

        self.use_fps = args.use_fps
        self.use_weights = args.use_weights

        self.detector_1 = KeypointDetector(nsample=1024, k=64, in_channels=0, out_channels=[32,32,64], fps=self.use_fps)
        self.detector_2 = KeypointDetector(nsample=512, k=32, in_channels=64, out_channels=[64,64,128], fps=self.use_fps)
        self.detector_3 = KeypointDetector(nsample=256, k=16, in_channels=128, out_channels=[128,128,256], fps=self.use_fps)

        if args.freeze_detector:
            for p in self.parameters():
                p.requires_grad = False
        
        self.desc_extractor_1 = DescExtractor(in_channels=0, out_channels=[32,32,64], C_detector=64, desc_dim=64)
        self.desc_extractor_2 = DescExtractor(in_channels=64, out_channels=[64,64,128], C_detector=128, desc_dim=128)
        self.desc_extractor_3 = DescExtractor(in_channels=128, out_channels=[128,128,256], C_detector=256, desc_dim=256)
    
    def forward(self, points):
        xyz_1, sigmas_1, attentive_feature_1, grouped_features_1, attentive_feature_map_1 = self.detector_1(points, None)
        desc_1 = self.desc_extractor_1(grouped_features_1, attentive_feature_map_1)
        if self.use_weights:
            weights_1 = 1.0/(sigmas_1 + 1e-5)
            weights_1_mean = torch.mean(weights_1, dim=1, keepdim=True)
            weights_1 = weights_1 / weights_1_mean
            xyz_2, sigmas_2, attentive_feature_2, grouped_features_2, attentive_feature_map_2 = self.detector_2(xyz_1, attentive_feature_1, weights_1)
            desc_2 = self.desc_extractor_2(grouped_features_2, attentive_feature_map_2)

            weights_2 = 1.0/(sigmas_2 + 1e-5)
            weights_2_mean = torch.mean(weights_2, dim=1, keepdim=True)
            weights_2 = weights_2 / weights_2_mean
            xyz_3, sigmas_3, attentive_feature_3, grouped_features_3, attentive_feature_map_3 = self.detector_3(xyz_2, attentive_feature_2, weights_2)
            desc_3 = self.desc_extractor_3(grouped_features_3, attentive_feature_map_3)
        else:
            xyz_2, sigmas_2, attentive_feature_2, grouped_features_2, attentive_feature_map_2 = self.detector_2(xyz_1, attentive_feature_1)
            desc_2 = self.desc_extractor_2(grouped_features_2, attentive_feature_map_2)
            xyz_3, sigmas_3, attentive_feature_3, grouped_features_3, attentive_feature_map_3 = self.detector_3(xyz_2, attentive_feature_2)
            desc_3 = self.desc_extractor_3(grouped_features_3, attentive_feature_map_3)
        
        ret_dict = {}
        ret_dict['xyz_1'] = xyz_1
        ret_dict['xyz_2'] = xyz_2
        ret_dict['xyz_3'] = xyz_3
        ret_dict['sigmas_1'] = sigmas_1
        ret_dict['sigmas_2'] = sigmas_2
        ret_dict['sigmas_3'] = sigmas_3
        ret_dict['desc_1'] = desc_1
        ret_dict['desc_2'] = desc_2
        ret_dict['desc_3'] = desc_3

        return ret_dict

class HRegNet(nn.Module):

    def __init__(self, args):
        super(HRegNet, self).__init__()
        self.feature_extraction = HierFeatureExtraction(args)

        # Freeze pretrained features when train
        if args.freeze_feats:
            for p in self.parameters():
                p.requires_grad = False
        
        self.coarse_corres = CoarseReg(k=8, in_channels=256, use_sim=True, use_neighbor=True)
        self.fine_corres_2 = FineReg(k=8, in_channels=128)
        self.fine_corres_1 = FineReg(k=8, in_channels=64)

        self.svd_head = WeightedSVDHead()
    
    def forward(self, src_points, dst_points):
        # Feature extraction
        src_feats = self.feature_extraction(src_points)
        dst_feats = self.feature_extraction(dst_points)

        # Coarse registration
        src_xyz_corres_3, src_dst_weights_3 = self.coarse_corres(src_feats['xyz_3'], src_feats['desc_3'], dst_feats['xyz_3'], \
            dst_feats['desc_3'], src_feats['sigmas_3'], dst_feats['sigmas_3'])
        
        R3, t3 = self.svd_head(src_feats['xyz_3'], src_xyz_corres_3, src_dst_weights_3)

        # Fine registration: Layer 2
        src_xyz_2_trans = torch.matmul(R3, src_feats['xyz_2'].permute(0,2,1).contiguous()) + t3.unsqueeze(2)
        src_xyz_2_trans = src_xyz_2_trans.permute(0,2,1).contiguous()
        src_xyz_corres_2, src_dst_weights_2 = self.fine_corres_2(src_xyz_2_trans, src_feats['desc_2'], dst_feats['xyz_2'], \
            dst_feats['desc_2'], src_feats['sigmas_2'], dst_feats['sigmas_2'])
        R2_, t2_ = self.svd_head(src_xyz_2_trans, src_xyz_corres_2, src_dst_weights_2)
        T3 = torch.zeros(R3.shape[0],4,4).cuda()
        T3[:,:3,:3] = R3
        T3[:,:3,3] = t3
        T3[:,3,3] = 1.0
        T2_ = torch.zeros(R2_.shape[0],4,4).cuda()
        T2_[:,:3,:3] = R2_
        T2_[:,:3,3] = t2_
        T2_[:,3,3] = 1.0
        T2 = torch.matmul(T2_, T3)
        R2 = T2[:,:3,:3]
        t2 = T2[:,:3,3]

        # Fine registration: Layer 1
        src_xyz_1_trans = torch.matmul(R2, src_feats['xyz_1'].permute(0,2,1).contiguous()) + t2.unsqueeze(2)
        src_xyz_1_trans = src_xyz_1_trans.permute(0,2,1).contiguous()
        src_xyz_corres_1, src_dst_weights_1 = self.fine_corres_1(src_xyz_1_trans, src_feats['desc_1'], dst_feats['xyz_1'], \
            dst_feats['desc_1'], src_feats['sigmas_1'], dst_feats['sigmas_1'])
        R1_, t1_ = self.svd_head(src_xyz_1_trans, src_xyz_corres_1, src_dst_weights_1)
        T1_ = torch.zeros(R1_.shape[0],4,4).cuda()
        T1_[:,:3,:3] = R1_
        T1_[:,:3,3] = t1_
        T1_[:,3,3] = 1.0

        T1 = torch.matmul(T1_, T2)
        R1 = T1[:,:3,:3]
        t1 = T1[:,:3,3]

        corres_dict = {}
        corres_dict['src_xyz_corres_3'] = src_xyz_corres_3
        corres_dict['src_xyz_corres_2'] = src_xyz_corres_2
        corres_dict['src_xyz_corres_1'] = src_xyz_corres_1
        corres_dict['src_dst_weights_3'] = src_dst_weights_3
        corres_dict['src_dst_weights_2'] = src_dst_weights_2
        corres_dict['src_dst_weights_1'] = src_dst_weights_1

        ret_dict = {}
        ret_dict['rotation'] = [R3, R2, R1]
        ret_dict['translation'] = [t3, t2, t1]
        ret_dict['src_feats'] = src_feats
        ret_dict['dst_feats'] = dst_feats

        return ret_dict

if __name__ == '__main__':
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser('HRegNet')

        parser.add_argument('--npoints', type=int, default=16384, help='number of input points')
        parser.add_argument('--freeze_detector', action='store_true')
        parser.add_argument('--use_fps', action='store_false')
        parser.add_argument('--freeze_features', action='store_true')
        parser.add_argument('--use_weights', action='store_true')

        return parser.parse_args()
    
    args = parse_args()
    args.use_fps = True
    args.use_weights = True
    model = HRegNet(args).cuda()
    xyz1 = torch.rand(2,16384,3).cuda()
    xyz2 = torch.rand(2,16384,3).cuda()
    ret_dict = model(xyz1, xyz2)
    print(ret_dict['rotation'][-1].shape)
    print(ret_dict['translation'][-1].shape)