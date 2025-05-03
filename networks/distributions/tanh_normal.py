
# import functools
# from typing import Optional, Type
# import flax.linen as nn
# import jax.numpy as jnp
# from networks.initialization import uniform_init
# import tensorflow_probability
# from networks.distributions.tanh_transformed import TanhTransformedDistribution

# # from jaxrl5.distributions.tanh_transformed import TanhTransformedDistribution

# tfp = tensorflow_probability.substrates.jax
# tfd = tfp.distributions


# class Normal(nn.Module):
#     base_cls: Type[nn.Module]
#     action_dim: int
#     log_std_min: Optional[float] = -10.
#     log_std_max: Optional[float] = 2.
#     state_dependent_std: bool = True
#     squash_tanh: bool = False

#     @nn.compact
#     def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
#         x = self.base_cls()(inputs, *args, **kwargs)

#         means = nn.Dense(
#             self.action_dim, kernel_init=uniform_init(), name="OutputDenseMean"
#         )(x)
#         if self.state_dependent_std:
#             log_stds = nn.Dense(
#                 self.action_dim, kernel_init=uniform_init(), name="OutputDenseLogStd"
#             )(x)
#         else:
#             log_stds = self.param(
#                 "OutputLogStd", nn.initializers.zeros, (self.action_dim,), jnp.float32
#             )

#         log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

#         distribution = tfd.MultivariateNormalDiag(
#             loc=means, scale_diag=jnp.exp(log_stds)
#         )

#         if self.squash_tanh:
#             return TanhTransformedDistribution(distribution)
#         else:
#             return distribution


# NormalTanh = functools.partial(Normal, squash_tanh=True)

import torch
import torch.nn as nn
import torch.distributions as td
from typing import Type, Optional
from functools import partial

class Normal(nn.Module):
    def __init__(
        self,
        base_cls: Type[nn.Module],
        action_dim: int,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        state_dependent_std: bool = True,
        squash_tanh: bool = False
    ):
        super().__init__()
        self.base_net = base_cls()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dependent_std = state_dependent_std
        self.squash_tanh = squash_tanh

        # 均值网络
        self.mean_layer = nn.Linear(self.base_net.output_dim, action_dim)
        self._init_weights(self.mean_layer)
        
        # 标准差网络
        if self.state_dependent_std:
            self.log_std_layer = nn.Linear(self.base_net.output_dim, action_dim)
            self._init_weights(self.log_std_layer)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _init_weights(self, layer):
        """初始化权重（对应JAX的uniform_init）"""
        nn.init.uniform_(layer.weight, -1e-3, 1e-3)
        nn.init.constant_(layer.bias, 0.0)

    def forward(self, inputs) -> td.Distribution:
        # 通过基础网络
        features = self.base_net(inputs)
        
        # 计算均值
        means = self.mean_layer(features)
        
        # 计算log标准差
        if self.state_dependent_std:
            log_stds = self.log_std_layer(features)
        else:
            log_stds = self.log_std.expand(inputs.size(0), -1)
        
        # 裁剪log标准差
        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        
        # 创建分布
        distribution = td.Independent(
            td.Normal(loc=means, scale=torch.exp(log_stds)),
            1
        )
        
        # 应用tanh变换
        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        return distribution

class TanhTransformedDistribution(td.TransformedDistribution):
    def __init__(self, base_distribution):
        super().__init__(
            base_distribution,
            td.transforms.TanhTransform()
        )

    def log_prob(self, actions):
        # 添加Jacobian修正
        log_prob = super().log_prob(actions)
        return log_prob - 2 * (torch.log(torch.tensor(2.0)) - 
               torch.log(1 - torch.tanh(actions).pow(2) + 1e-6)).sum(dim=-1)

# 快捷方式定义
NormalTanh = partial(Normal, squash_tanh=True)