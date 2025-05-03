
# from typing import Any, Optional
# import jax.numpy as jnp
# import tensorflow_probability

# tfp = tensorflow_probability.substrates.jax
# tfd = tfp.distributions
# tfb = tfp.bijectors


# # Inspired by
# # https://github.com/deepmind/acme/blob/300c780ffeb88661a41540b99d3e25714e2efd20/acme/jax/networks/distributional.py#L163
# # but modified to only compute a mode.


# class TanhTransformedDistribution(tfd.TransformedDistribution):
#     def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
#         super().__init__(
#             distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
#         )

#     def mode(self) -> jnp.ndarray:
#         return self.bijector.forward(self.distribution.mode())

#     @classmethod
#     def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
#         td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
#         del td_properties["bijector"]
#         return td_properties

import torch
import torch.distributions as td

class TanhTransformedDistribution(td.TransformedDistribution):
    def __init__(self, base_distribution: td.Distribution, validate_args: bool = False):
        super().__init__(
            base_distribution,
            [td.TanhTransform()],  # 将Tanh变换作为转换链
            validate_args=validate_args
        )
    
    def mode(self) -> torch.Tensor:
        """计算变换后的分布模式"""
        # 获取基础分布的mode
        base_mode = self.base_dist.mode()
        # 应用所有注册的变换（这里只有Tanh变换）
        for transform in self.transforms:
            base_mode = transform(base_mode)
        return base_mode