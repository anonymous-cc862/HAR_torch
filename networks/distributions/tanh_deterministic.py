# from typing import Type

# import flax.linen as nn
# import jax.numpy as jnp

# from networks.initialization import uniform_init


# class TanhDeterministic(nn.Module):
#     base_cls: Type[nn.Module]
#     action_dim: int

#     @nn.compact
#     def __call__(self, inputs, *args, **kwargs) -> jnp.ndarray:
#         x = self.base_cls()(inputs, *args, **kwargs)

#         means = nn.Dense(
#             self.action_dim, kernel_init=uniform_init(), name="OutputDenseMean"
#         )(x)

#         means = nn.tanh(means)

#         return means

import torch
import torch.nn as nn
from typing import Type

class TanhDeterministic(nn.Module):
    def __init__(self, base_cls: Type[nn.Module], action_dim: int):
        super().__init__()
        self.base_net = base_cls()  # 实例化基础网络
        self.dense_mean = nn.Linear(in_features=..., out_features=action_dim)  # 需要设置正确输入维度
        
        # 初始化权重 (假设uniform_init对应PyTorch的均匀初始化)
        nn.init.uniform_(self.dense_mean.weight, a=-0.01, b=0.01)  # 示例初始化范围
        nn.init.zeros_(self.dense_mean.bias)

    def forward(self, inputs) -> torch.Tensor:
        # 基础网络前向传播
        x = self.base_net(inputs)
        
        # 生成均值
        means = self.dense_mean(x)
        
        # 应用tanh激活
        return torch.tanh(means)