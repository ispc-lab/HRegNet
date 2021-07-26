import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import furthest_point_sample, weighted_furthest_point_sample, gather_operation
from pytorch3d.ops import knn_points, knn_gather

def knn_group(xyz1, xyz2, features2, k):
    '''
    Input:
        xyz1: query points, [B,M,3] 
        xyz2: database points, [B,N,3]
        features2: [B,C,N]
        k: int
    Output:
        grouped_features: [B,4+C,M,k]
        knn_xyz: [B,M,k,3]
    '''
    _, knn_idx, knn_xyz = knn_points(xyz1, xyz2, K=k, return_nn=True)
    rela_xyz = knn_xyz - xyz1.unsqueeze(2).repeat(1,1,k,1) # [B,M,k,3]
    rela_dist = torch.norm(rela_xyz, dim=-1, keepdim=True) # [B,M,k,1]
    grouped_features =  torch.cat([rela_xyz,rela_dist], dim=-1)
    if features2 is not None:
        knn_features = knn_gather(features2.permute(0,2,1).contiguous(), knn_idx)
        grouped_features = torch.cat([rela_xyz,rela_dist,knn_features],dim=-1) # [B,M,k,4+C]
    return grouped_features.permute(0,3,1,2).contiguous(), knn_xyz

def calc_cosine_similarity(desc1, desc2):
    '''
    Input:
        desc1: [B,N,*,C]
        desc2: [B,N,*,C]
    Ret:
        similarity: [B,N,*]
    '''
    inner_product = torch.sum(torch.mul(desc1, desc2), dim=-1, keepdim=False)
    norm_1 = torch.norm(desc1, dim=-1, keepdim=False)
    norm_2 = torch.norm(desc2, dim=-1, keepdim=False)
    similarity = inner_product/(torch.mul(norm_1, norm_2)+1e-6)
    return similarity

class KeypointDetector(nn.Module):
    '''
    Params:
        nsample: number of sampled points
        k: k nearest neighbors
        in_channels: input channel number
        out_channels: output channel number
        fps: use furthest point sampling
    Input:
        xyz: [B,N,3]
        features: [B,N,C_in]
        weights: None / [B,N]
    Output:
        keypoints: [B,M,3]
        weights: [B,M]
        attentive_feature: [B,C_o,M]
        grouped_features: [B,C_in+4,M,k]
        attentive_feature_map: [B,C_o,M,k]
    '''
    def __init__(self, nsample, k, in_channels, out_channels, fps=True):
        super(KeypointDetector, self).__init__()

        self.nsample = nsample
        self.k = k
        self.fps = fps

        layers = []
        out_channels = [in_channels+4, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
                       nn.BatchNorm2d(out_channels[i]),
                       nn.ReLU()]
        self.convs = nn.Sequential(*layers)
        self.C_o1 = out_channels[-1]

        self.mlp1 = nn.Sequential(nn.Conv1d(self.C_o1, self.C_o1, kernel_size=1),
                                  nn.BatchNorm1d(self.C_o1),
                                  nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(self.C_o1, self.C_o1, kernel_size=1),
                                  nn.BatchNorm1d(self.C_o1),
                                  nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(self.C_o1, 1, kernel_size=1))

        self.softplus = nn.Softplus()
    
    def forward(self, xyz, features, weights=None):
        # Use FPS or random sampling
        if self.fps:
            # Use FPS or WFPS
            if weights is None:
                fps_idx = furthest_point_sample(xyz, self.nsample)
                sampled_xyz = gather_operation(xyz.permute(0,2,1).contiguous(), fps_idx).permute(0,2,1).contiguous()
            else:
                fps_idx = weighted_furthest_point_sample(xyz, weights, self.nsample)
                sampled_xyz = gather_operation(xyz.permute(0,2,1).contiguous(), fps_idx).permute(0,2,1).contiguous()
        else:
            N = xyz.shape[1]
            rand_idx = torch.randperm(N)[:self.nsample]
            sampled_xyz = xyz[:,rand_idx,:]
        
        grouped_features, knn_xyz = knn_group(sampled_xyz, xyz, features, self.k) # [B,4+C1,M,k] [B,M,k,3]
        embedding = self.convs(grouped_features)
        x1 = torch.max(embedding, dim=1, keepdim=False)[0] # [B,M,k]
        attentive_weights = F.softmax(x1, dim=-1) # [B,M,k]

        weights_xyz = attentive_weights.unsqueeze(-1).repeat(1,1,1,3)
        keypoints = torch.sum(torch.mul(weights_xyz, knn_xyz),dim=2,keepdim=False) # [B,M,3]

        weights_feature = attentive_weights.unsqueeze(1).repeat(1,self.C_o1,1,1)
        attentive_feature_map = torch.mul(embedding, weights_feature) # [B,C2,M,k]
        attentive_feature = torch.sum(attentive_feature_map, dim=-1, keepdim=False)

        sigmas = self.mlp3(self.mlp2(self.mlp1(attentive_feature)))
        sigmas = self.softplus(sigmas) + 0.001
        sigmas = sigmas.squeeze(1)

        return keypoints, sigmas, attentive_feature, grouped_features, attentive_feature_map

class DescExtractor(nn.Module):
    '''
    Params:
        in_channels: input channel number
        out_channels: output channel number
        C_detector: channel number of keypoint detector (attentive feature map)
        desc_dim: dimension of descriptor
    Input:
        grouped_features: [B,C_in+4,M,k]
        attentive_feature_map: [B,C_detector,M,k]
    Output:
        desc: [B,desc_dim,M]
    '''
    def __init__(self, in_channels, out_channels, C_detector, desc_dim):
        super(DescExtractor, self).__init__()

        layers = []
        out_channels = [in_channels+4, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
                       nn.BatchNorm2d(out_channels[i]),
                       nn.ReLU()]
        self.convs = nn.Sequential(*layers)
        
        self.C_o1 = out_channels[-1]
        
        self.mlp1 = nn.Sequential(nn.Conv2d(2*self.C_o1+C_detector, out_channels[-2], kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels[-2]),
                                   nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv2d(out_channels[-2], desc_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(desc_dim),
                                   nn.ReLU())
    
    def forward(self, grouped_features, attentive_feature_map):
        x1 = self.convs(grouped_features)
        x2 = torch.max(x1, dim=3, keepdim=True)[0]
        k = x1.shape[-1]
        x2 = x2.repeat(1,1,1,k)
        x2 = torch.cat((x2, x1),dim=1) # [B,2*C_o1,N,k]
        x2 = torch.cat((x2, attentive_feature_map), dim=1)
        x2 = self.mlp2(self.mlp1(x2))
        desc = torch.max(x2, dim=3, keepdim=False)[0]
        return desc

class CoarseReg(nn.Module):
    '''
    Params:
        k: number of candidate keypoints
        in_channels: input channel number
        use_sim: use original similarity features
        use_neighbor: use neighbor aware similarity features
    Input:
        src_xyz: [B,N,3]
        src_desc: [B,C,N]
        dst_xyz: [B,N,3]
        dst_desc: [B,C,N]
        src_weights: [B,N]
        dst_weights: [B,N]
    Output:
        corres_xyz: [B,N,3]
        weights: [B,N]
    '''
    def __init__(self, k, in_channels, use_sim=True, use_neighbor=True):
        super(CoarseReg, self).__init__()

        self.k = k

        self.use_sim = use_sim
        self.use_neighbor = use_neighbor

        if self.use_sim and self.use_neighbor:
            out_channels = [in_channels*2+16, in_channels*2, in_channels*2, in_channels*2]
        elif self.use_sim:
            out_channels = [in_channels*2+14, in_channels*2, in_channels*2, in_channels*2]
        elif self.use_neighbor:
            out_channels = [in_channels*2+14, in_channels*2, in_channels*2, in_channels*2]
        else:
            out_channels = [in_channels*2+12, in_channels*2, in_channels*2, in_channels*2]
        
        layers = []

        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
                       nn.BatchNorm2d(out_channels[i]),
                       nn.ReLU()]
        self.convs_1 = nn.Sequential(*layers)

        out_channels_nbr = [in_channels+4, in_channels, in_channels, in_channels]
        self_layers = []
        for i in range(1, len(out_channels_nbr)):
            self_layers += [nn.Conv2d(out_channels_nbr[i-1], out_channels_nbr[i], kernel_size=1, bias=False),
                       nn.BatchNorm2d(out_channels_nbr[i]),
                       nn.ReLU()]
        self.convs_2 = nn.Sequential(*self_layers)

        self.mlp1 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
                                  nn.BatchNorm1d(in_channels*2),
                                  nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
                                  nn.BatchNorm1d(in_channels*2),
                                  nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(in_channels*2, 1, kernel_size=1))

    def forward(self, src_xyz, src_desc, dst_xyz, dst_desc, src_weights, dst_weights):
        src_desc = src_desc.permute(0,2,1).contiguous()
        dst_desc = dst_desc.permute(0,2,1).contiguous()
        _, src_knn_idx, src_knn_desc = knn_points(src_desc, dst_desc, K=self.k, return_nn=True)
        src_knn_xyz = knn_gather(dst_xyz, src_knn_idx) # [B,N,k,3]
        src_xyz_expand = src_xyz.unsqueeze(2).repeat(1,1,self.k,1)
        src_desc_expand = src_desc.unsqueeze(2).repeat(1,1,self.k,1) # [B,N,k,C]
        src_rela_xyz = src_knn_xyz - src_xyz_expand # [B,N,k,3]
        src_rela_dist = torch.norm(src_rela_xyz, dim=-1, keepdim=True) # [B,N,k,1]
        src_weights_expand = src_weights.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.k,1) # [B,N,k,1]
        src_knn_weights = knn_gather(dst_weights.unsqueeze(-1), src_knn_idx) # [B,N,k,1]

        if self.use_sim:
            # construct original similarity features
            dst_desc_expand_N = dst_desc.unsqueeze(2).repeat(1,1,src_xyz.shape[1],1) # [B,N2,N1,C]
            src_desc_expand_N = src_desc.unsqueeze(1).repeat(1,dst_xyz.shape[1],1,1) # [B,N2,N1,C]

            dst_src_cos = calc_cosine_similarity(dst_desc_expand_N, src_desc_expand_N) # [B,N2,N1]
            dst_src_cos_max = torch.max(dst_src_cos, dim=2, keepdim=True)[0] # [B,N2,1]
            dst_src_cos_norm = dst_src_cos/(dst_src_cos_max+1e-6) # [B,N2,N1]

            src_dst_cos = dst_src_cos.permute(0,2,1) # [B,N1,N2]
            src_dst_cos_max = torch.max(src_dst_cos, dim=2, keepdim=True)[0] # [B,N1,1]
            src_dst_cos_norm = src_dst_cos/(src_dst_cos_max+1e-6) # [B,N1,N2]
            
            dst_src_cos_knn = knn_gather(dst_src_cos_norm, src_knn_idx) # [B,N1,k,N1]
            dst_src_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
                src_knn_xyz.shape[2]).cuda() # [B,N1,k]
            for i in range(src_xyz.shape[1]):
                dst_src_cos[:,i,:] = dst_src_cos_knn[:,i,:,i]
            
            src_dst_cos_knn = knn_gather(src_dst_cos_norm.permute(0,2,1), src_knn_idx)
            src_dst_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
                src_knn_xyz.shape[2]).cuda() # [B,N1,k]
            for i in range(src_xyz.shape[1]):
                src_dst_cos[:,i,:] = src_dst_cos_knn[:,i,:,i]
        
        if self.use_neighbor:
            _, src_nbr_knn_idx, src_nbr_knn_xyz = knn_points(src_xyz, src_xyz, K=self.k, return_nn=True)
            src_nbr_knn_feats = knn_gather(src_desc, src_nbr_knn_idx) # [B,N,k,C]
            src_nbr_knn_rela_xyz = src_nbr_knn_xyz - src_xyz_expand # [B,N,k,3]
            src_nbr_knn_rela_dist = torch.norm(src_nbr_knn_rela_xyz, dim=-1, keepdim=True) # [B,N,k]
            src_nbr_feats = torch.cat([src_nbr_knn_feats, src_nbr_knn_rela_xyz, src_nbr_knn_rela_dist], dim=-1)

            _, dst_nbr_knn_idx, dst_nbr_knn_xyz = knn_points(dst_xyz, dst_xyz, K=self.k, return_nn=True)
            dst_nbr_knn_feats = knn_gather(dst_desc, dst_nbr_knn_idx) # [B,N,k,C]
            dst_xyz_expand = dst_xyz.unsqueeze(2).repeat(1,1,self.k,1)
            dst_nbr_knn_rela_xyz = dst_nbr_knn_xyz - dst_xyz_expand # [B,N,k,3]
            dst_nbr_knn_rela_dist = torch.norm(dst_nbr_knn_rela_xyz, dim=-1, keepdim=True) # [B,N,k]
            dst_nbr_feats = torch.cat([dst_nbr_knn_feats, dst_nbr_knn_rela_xyz, dst_nbr_knn_rela_dist], dim=-1)

            src_nbr_weights = self.convs_2(src_nbr_feats.permute(0,3,1,2).contiguous())
            src_nbr_weights = torch.max(src_nbr_weights, dim=1, keepdim=False)[0]
            src_nbr_weights = F.softmax(src_nbr_weights, dim=-1)
            src_nbr_desc = torch.sum(torch.mul(src_nbr_knn_feats, src_nbr_weights.unsqueeze(-1)),dim=2, keepdim=False)

            dst_nbr_weights = self.convs_2(dst_nbr_feats.permute(0,3,1,2).contiguous())
            dst_nbr_weights = torch.max(dst_nbr_weights, dim=1, keepdim=False)[0]
            dst_nbr_weights = F.softmax(dst_nbr_weights, dim=-1)
            dst_nbr_desc = torch.sum(torch.mul(dst_nbr_knn_feats, dst_nbr_weights.unsqueeze(-1)),dim=2, keepdim=False)

            dst_nbr_desc_expand_N = dst_nbr_desc.unsqueeze(2).repeat(1,1,src_xyz.shape[1],1) # [B,N2,N1,C]
            src_nbr_desc_expand_N = src_nbr_desc.unsqueeze(1).repeat(1,dst_xyz.shape[1],1,1) # [B,N2,N1,C]

            dst_src_nbr_cos = calc_cosine_similarity(dst_nbr_desc_expand_N, src_nbr_desc_expand_N) # [B,N2,N1]
            dst_src_nbr_cos_max = torch.max(dst_src_nbr_cos, dim=2, keepdim=True)[0] # [B,N2,1]
            dst_src_nbr_cos_norm = dst_src_nbr_cos/(dst_src_nbr_cos_max+1e-6) # [B,N2,N1]

            src_dst_nbr_cos = dst_src_nbr_cos.permute(0,2,1) # [B,N1,N2]
            src_dst_nbr_cos_max = torch.max(src_dst_nbr_cos, dim=2, keepdim=True)[0] # [B,N1,1]
            src_dst_nbr_cos_norm = src_dst_nbr_cos/(src_dst_nbr_cos_max+1e-6) # [B,N1,N2]
            
            dst_src_nbr_cos_knn = knn_gather(dst_src_nbr_cos_norm, src_knn_idx) # [B,N1,k,N1]
            dst_src_nbr_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
                src_knn_xyz.shape[2]).to(src_knn_xyz.cuda()) # [B,N1,k]
            for i in range(src_xyz.shape[1]):
                dst_src_nbr_cos[:,i,:] = dst_src_nbr_cos_knn[:,i,:,i]
            
            src_dst_nbr_cos_knn = knn_gather(src_dst_nbr_cos_norm.permute(0,2,1), src_knn_idx)
            src_dst_nbr_cos = torch.zeros(src_knn_xyz.shape[0], src_knn_xyz.shape[1], \
                src_knn_xyz.shape[2]).to(src_knn_xyz.cuda()) # [B,N1,k]
            for i in range(src_xyz.shape[1]):
                src_dst_nbr_cos[:,i,:] = src_dst_nbr_cos_knn[:,i,:,i]
            
        geom_feats = torch.cat([src_rela_xyz, src_rela_dist, src_xyz_expand, src_knn_xyz],dim=-1)
        desc_feats = torch.cat([src_desc_expand, src_knn_desc, src_weights_expand, src_knn_weights],dim=-1)
        if self.use_sim and self.use_neighbor:
            similarity_feats = torch.cat([src_dst_cos.unsqueeze(-1), dst_src_cos.unsqueeze(-1), \
                src_dst_nbr_cos.unsqueeze(-1), dst_src_nbr_cos.unsqueeze(-1)], dim=-1)
        elif self.use_sim:
            similarity_feats = torch.cat([src_dst_cos.unsqueeze(-1), dst_src_cos.unsqueeze(-1)],dim=-1)
        elif self.use_neighbor:
            similarity_feats = torch.cat([src_dst_nbr_cos.unsqueeze(-1), dst_src_nbr_cos.unsqueeze(-1)], dim=-1)
        else:
            similarity_feats = None
        
        if self.use_sim or self.use_neighbor:
            feats = torch.cat([geom_feats, desc_feats, similarity_feats],dim=-1)
        else:
            feats = torch.cat([geom_feats, desc_feats],dim=-1)
        
        feats = self.convs_1(feats.permute(0,3,1,2))
        attentive_weights = torch.max(feats, dim=1)[0]
        attentive_weights = F.softmax(attentive_weights, dim=-1) # [B,N,k]
        corres_xyz = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), src_knn_xyz), dim=2, keepdim=False) # [B,N,3]
        attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(1), feats), dim=-1, keepdim=False) # [B,N,C]
        weights = self.mlp3(self.mlp2(self.mlp1(attentive_feats))) # [B,1,N]
        weights = torch.sigmoid(weights.squeeze(1))

        return corres_xyz, weights

class FineReg(nn.Module):
    '''
    Params:
        k: number of candidate keypoints
        in_channels: input channel number
    Input:
        src_xyz: [B,N,3]
        src_desc: [B,C,N]
        dst_xyz: [B,N,3]
        dst_desc: [B,C,N]
        src_weights: [B,N]
        dst_weights: [B,N]
    Output:
        corres_xyz: [B,N,3]
        weights: [B,N]
    '''
    def __init__(self, k, in_channels):
        super(FineReg, self).__init__()
        self.k = k
        out_channels = [in_channels*2+12, in_channels*2, in_channels*2, in_channels*2]
        layers = []
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=1, bias=False),
                       nn.BatchNorm2d(out_channels[i]),
                       nn.ReLU()]
        self.convs_1 = nn.Sequential(*layers)

        self.mlp1 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
                                  nn.BatchNorm1d(in_channels*2),
                                  nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(in_channels*2, in_channels*2, kernel_size=1),
                                  nn.BatchNorm1d(in_channels*2),
                                  nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(in_channels*2, 1, kernel_size=1))
    
    def forward(self, src_xyz, src_feat, dst_xyz, dst_feat, src_weights, dst_weights):
        _, src_knn_idx, src_knn_xyz = knn_points(src_xyz, dst_xyz, K=self.k, return_nn=True)
        src_feat = src_feat.permute(0,2,1).contiguous()
        dst_feat = dst_feat.permute(0,2,1).contiguous()
        src_knn_feat = knn_gather(dst_feat, src_knn_idx) # [B,N,k,C]
        src_xyz_expand = src_xyz.unsqueeze(2).repeat(1,1,self.k,1)
        src_feat_expand = src_feat.unsqueeze(2).repeat(1,1,self.k,1)
        src_rela_xyz = src_knn_xyz - src_xyz_expand # [B,N,k,3]
        src_rela_dist = torch.norm(src_rela_xyz, dim=-1, keepdim=True) # [B,N,k,1]
        src_weights_expand = src_weights.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.k,1) # [B,N,k,1]
        src_knn_weights = knn_gather(dst_weights.unsqueeze(-1), src_knn_idx) # [B,N,k,1]
        feats = torch.cat([src_rela_xyz, src_rela_dist, src_xyz_expand, src_knn_xyz, \
            src_feat_expand, src_knn_feat, src_weights_expand, src_knn_weights], dim=-1)
        feats = self.convs_1(feats.permute(0,3,1,2).contiguous()) # [B,C,N,k]
        attentive_weights = torch.max(feats, dim=1)[0]
        attentive_weights = F.softmax(attentive_weights, dim=-1) # [B,N,k]
        corres_xyz = torch.sum(torch.mul(attentive_weights.unsqueeze(-1), src_knn_xyz), dim=2, keepdim=False) # [B,N,3]
        attentive_feats = torch.sum(torch.mul(attentive_weights.unsqueeze(1), feats), dim=-1, keepdim=False) # [B,N,C]
        weights = self.mlp3(self.mlp2(self.mlp1(attentive_feats))) # [B,1,N]
        weights = torch.sigmoid(weights.squeeze(1))

        return corres_xyz, weights

class WeightedSVDHead(nn.Module):
    '''
    Input:
        src: [B,N,3]
        src_corres: [B,N,3]
        weights: [B,N]
    Output:
        r: [B,3,3]
        t: [B,3]
    '''
    def __init__(self):
        super(WeightedSVDHead, self).__init__()
    
    def forward(self, src, src_corres, weights):
        eps = 1e-4
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = weights/sum_weights
        weights = weights.unsqueeze(2)

        src_mean = torch.matmul(weights.transpose(1,2),src)/(torch.sum(weights,dim=1).unsqueeze(1)+eps)
        src_corres_mean = torch.matmul(weights.transpose(1,2),src_corres)/(torch.sum(weights,dim=1).unsqueeze(1)+eps)

        src_centered = src - src_mean # [B,N,3]
        src_corres_centered = src_corres - src_corres_mean # [B,N,3]

        weight_matrix = torch.diag_embed(weights.squeeze(2))
        
        cov_mat = torch.matmul(src_centered.transpose(1,2),torch.matmul(weight_matrix,src_corres_centered))

        try:
            u, s, v = torch.svd(cov_mat)
        except Exception as e:
            r = torch.eye(3).cuda()
            r = r.repeat(src_mean.shape[0],1,1)
            t = torch.zeros((src_mean.shape[0],3,1)).cuda()
            t = t.view(t.shape[0], 3)

            return r, t
        
        tm_determinant = torch.det(torch.matmul(v.transpose(1,2), u.transpose(1,2)))
        
        determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0], 2)).cuda(),tm_determinant.unsqueeze(1)), 1))

        r = torch.matmul(v, torch.matmul(determinant_matrix, u.transpose(1,2)))

        t = src_corres_mean.transpose(1,2) - torch.matmul(r, src_mean.transpose(1,2))
        t = t.view(t.shape[0], 3)
        
        return r, t