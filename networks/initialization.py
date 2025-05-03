
# import flax.linen as nn
# from jax import numpy as jnp
# from typing import Optional


# def orthogonal_init(scale: Optional[float] = None):
#     if scale is None:
#         scale = jnp.sqrt(2)
#     return nn.initializers.orthogonal(scale)  # , nn.initializers.normal(scale)


# def uniform_init(scale_final=None):
#     if scale_final is not None:
#         return nn.initializers.xavier_uniform(scale_final)
#     return nn.initializers.xavier_uniform()


# default_init = uniform_init


import torch
import torch.nn as nn
import math

def orthogonal_init(tensor: torch.Tensor, scale: Optional[float] = None):
    """
    PyTorch正交初始化（匹配JAX的orthogonal初始化行为）
    :param tensor: 需要初始化的张量（至少2维）
    :param scale: 缩放因子（默认sqrt(2)）
    """
    if scale is None:
        scale = math.sqrt(2)
    nn.init.orthogonal_(tensor, gain=scale)

def uniform_init(tensor: torch.Tensor, scale_final: Optional[float] = None):
    """
    PyTorch均匀初始化（匹配JAX的xavier_uniform行为）
    :param tensor: 需要初始化的张量
    :param scale_final: 最终缩放因子（对应JAX的scale参数）
    """
    gain = scale_final if scale_final is not None else 1.0
    nn.init.xavier_uniform_(tensor, gain=gain)

# 默认初始化方法
default_init = uniform_init

# 使用示例
if __name__ == "__main__":
    # 初始化线性层权重
    linear_layer = nn.Linear(256, 128)
    
    # 应用正交初始化
    orthogonal_init(linear_layer.weight)  # 默认使用scale=sqrt(2)
    
    # 应用带缩放的均匀初始化
    uniform_init(linear_layer.bias, scale_final=0.1)
    
    # # 验证初始化结果
    # print("权重均值:", linear_layer.weight.mean().item())
    # print("权重标准差:", linear_layer.weight.std().item())
    # print("偏置均值:", linear_layer.bias.mean().item())