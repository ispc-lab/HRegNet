import torch
import numpy as np
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather

def prob_chamfer_loss(keypoints1, keypoints2, sigma1, sigma2, gt_R, gt_t):
    """
    Calculate probabilistic chamfer distance between keypoints1 and keypoints2
    Input:
        keypoints1: [B,M,3]
        keypoints2: [B,M,3]
        sigma1: [B,M]
        sigma2: [B,M]
        gt_R: [B,3,3]
        gt_t: [B,3]
    """
    keypoints1 = keypoints1.permute(0,2,1).contiguous()
    keypoints1 = torch.matmul(gt_R, keypoints1) + gt_t.unsqueeze(2)
    keypoints2 = keypoints2.permute(0,2,1).contiguous()
    B, M = keypoints1.size()[0], keypoints1.size()[2]
    N = keypoints2.size()[2]

    keypoints1_expanded = keypoints1.unsqueeze(3).expand(B,3,M,N)
    keypoints2_expanded = keypoints2.unsqueeze(2).expand(B,3,M,N)

    # diff: [B, M, M]
    diff = torch.norm(keypoints1_expanded-keypoints2_expanded, dim=1, keepdim=False)

    if sigma1 is None or sigma2 is None:
        min_dist_forward, _ = torch.min(diff, dim=2, keepdim=False)
        forward_loss = min_dist_forward.mean()

        min_dist_backward, _ = torch.min(diff, dim=1, keepdim=False)
        backward_loss = min_dist_backward.mean()

        loss = forward_loss + backward_loss
    
    else:
        min_dist_forward, min_dist_forward_I = torch.min(diff, dim=2, keepdim=False)
        selected_sigma_2 = torch.gather(sigma2, dim=1, index=min_dist_forward_I)
        sigma_forward = (sigma1 + selected_sigma_2)/2
        forward_loss = (torch.log(sigma_forward)+min_dist_forward/sigma_forward).mean()

        min_dist_backward, min_dist_backward_I = torch.min(diff, dim=1, keepdim=False)
        selected_sigma_1 = torch.gather(sigma1, dim=1, index=min_dist_backward_I)
        sigma_backward = (sigma2 + selected_sigma_1)/2
        backward_loss = (torch.log(sigma_backward)+min_dist_backward/sigma_backward).mean()

        loss = forward_loss + backward_loss
    return loss

def matching_loss(src_kp, src_sigma, src_desc, dst_kp, dst_sigma, dst_desc, gt_R, gt_t, temp=0.1, sigma_max=3.0):
    src_kp = src_kp.permute(0,2,1).contiguous()
    src_kp = torch.matmul(gt_R, src_kp) + gt_t.unsqueeze(2)
    dst_kp = dst_kp.permute(0,2,1).contiguous()

    src_desc = src_desc.unsqueeze(3) # [B,C,M,1]
    dst_desc = dst_desc.unsqueeze(2) # [B,C,1,M]

    desc_dists = torch.norm((src_desc - dst_desc), dim=1) # [B,M,M]
    desc_dists_inv = 1.0/(desc_dists + 1e-3)
    desc_dists_inv = desc_dists_inv/temp

    score_src = F.softmax(desc_dists_inv, dim=2)
    score_dst = F.softmax(desc_dists_inv, dim=1).permute(0,2,1)

    src_kp = src_kp.permute(0,2,1)
    dst_kp = dst_kp.permute(0,2,1)

    src_kp_corres = torch.matmul(score_src, dst_kp)
    dst_kp_corres = torch.matmul(score_dst, src_kp)

    diff_forward = torch.norm((src_kp - src_kp_corres), dim=-1)
    diff_backward = torch.norm((dst_kp - dst_kp_corres), dim=-1)

    src_weights = torch.clamp(sigma_max - src_sigma, min=0.01)
    src_weights_mean = torch.mean(src_weights, dim=1, keepdim=True)
    src_weights = (src_weights/src_weights_mean).detach()

    dst_weights = torch.clamp(sigma_max - dst_sigma, min=0.01)
    dst_weights_mean = torch.mean(dst_weights, dim=1, keepdim=True)
    dst_weights = (dst_weights/dst_weights_mean).detach()

    loss_forward = (src_weights * diff_forward).mean()
    loss_backward = (dst_weights * diff_backward).mean()

    loss = loss_forward + loss_backward

    return loss

def transformation_loss(pred_R, pred_t, gt_R, gt_t, alpha=1.0):
    '''
    Input:
        pred_R: [B,3,3]
        pred_t: [B,3]
        gt_R: [B,3,3]
        gt_t: [B,3]
        alpha: weight
    '''
    Identity = []
    for i in range(pred_R.shape[0]):
        Identity.append(torch.eye(3,3).cuda())
    Identity = torch.stack(Identity, dim=0)
    resi_R = torch.norm((torch.matmul(pred_R.transpose(2,1).contiguous(),gt_R) - Identity), dim=(1,2), keepdim=False)
    resi_t = torch.norm((pred_t - gt_t), dim=1, keepdim=False)
    loss_R = torch.mean(resi_R)
    loss_t = torch.mean(resi_t)
    loss = alpha * loss_R + loss_t

    return loss, loss_R, loss_t