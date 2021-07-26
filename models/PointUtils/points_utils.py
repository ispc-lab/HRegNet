import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn as nn

from typing import Union

import point_utils_cuda

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        '''
        ctx:
        xyz: [B,N,3]
        npoint: int
        '''
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        point_utils_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output
    
    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthest_point_sample = FurthestPointSampling.apply

class WeightedFurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, weights: torch.Tensor, npoint: int) -> torch.Tensor:
        '''
        ctx:
        xyz: [B,N,3]
        weights: [B,N]
        npoint: int
        '''
        assert xyz.is_contiguous()
        assert weights.is_contiguous()
        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        point_utils_cuda.weighted_furthest_point_sampling_wrapper(B, N, npoint, xyz, weights, temp, output);
        return output
    
    @staticmethod
    def backward(xyz, a=None):
        return None, None

weighted_furthest_point_sample = WeightedFurthestPointSampling.apply

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        '''
        ctx
        features: [B,C,N]
        idx: [B,npoint]
        '''
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint)

        point_utils_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output
    
    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()
        grad_features = Variable(torch.cuda.FloatTensor(B,C,N).zero_())
        grad_out_data = grad_out.data.contiguous()
        point_utils_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None

gather_operation = GatherOperation.apply
