
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# from typing import Callable, Optional, Sequence, Type
# from networks.initialization import default_init


# class MLP(nn.Module):
#     hidden_dims: Sequence[int]
#     activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
#     activate_final: int = False
#     layer_norm: bool = False
#     dropout_rate: Optional[float] = None

#     @nn.compact
#     def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
#         for i, size in enumerate(self.hidden_dims):
#             x = nn.Dense(size, kernel_init=default_init())(x)
#             if i + 1 < len(self.hidden_dims) or self.activate_final:  # hidden layers
#                 if self.layer_norm:
#                     x = nn.LayerNorm()(x)
#                 x = self.activations(x)
#                 if self.dropout_rate is not None and self.dropout_rate > 0:
#                     x = nn.Dropout(rate=self.dropout_rate)(
#                         x, deterministic=not training)
#         return x


# # class TimeMLP(nn.Module):
# #     mlp: Type[nn.Module]
# #     time_embedding: Type[nn.Module]
# #     time_processor: Type[nn.Module]
# #     """
# #     if cond_embedding:
# #         treat the last dim as the conditional tag
# #     """
# #     @nn.compact
# #     def __call__(self,
# #                  x: jnp.ndarray,  # s = [obs, cond] if it's conditional
# #                  time: jnp.ndarray,
# #                  training: bool = False):
# #         t_ff = self.time_embedding()(time)
# #         time_suffix = self.time_processor()(t_ff, training=training)
# #         input_with_time = jnp.concatenate([x, time_suffix], axis=-1)
# #         return self.mlp()(input_with_time, training=training)
# #


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Sequence, List

class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int],
        activations: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        activate_final: bool = False,
        layer_norm: bool = False,
        dropout_rate: Optional[float] = None
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.LazyLinear(hidden_dims[i]))  # 自动推断输入维度
            
            # 添加层归一化
            if layer_norm and (i < len(hidden_dims)-1 or activate_final):
                layers.append(nn.LayerNorm(hidden_dims[i]))
                
            # 添加激活函数
            if (i < len(hidden_dims)-1) or activate_final:
                layers.append(nn.Identity())  # 占位符，实际激活函数在forward中应用
            
            # 添加Dropout
            if dropout_rate and dropout_rate > 0 and (i < len(hidden_dims)-1):
                layers.append(nn.Dropout(dropout_rate))
        
        self.layers = nn.ModuleList(layers)
        self._init_weights()

    def _init_weights(self):
        """使用Xavier均匀初始化"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            # 处理线性层
            if isinstance(layer, nn.Linear):
                x = layer(x)
            
            # 处理层归一化
            elif isinstance(layer, nn.LayerNorm):
                x = layer(x)
            
            # 处理激活函数占位符
            elif isinstance(layer, nn.Identity):
                # 判断是否需要激活
                if (i//3 < len(self.hidden_dims)-1) or self.activate_final:
                    x = self.activations(x)
            
            # 处理Dropout
            elif isinstance(layer, nn.Dropout):
                if training:
                    x = layer(x)
        return x

# class TimeMLP(nn.Module):
#     """
#     Time-conditioned MLP模块
#     示例用法：
#     time_mlp = TimeMLP(
#         mlp_class=MLP,
#         time_embedding=TimeEmbedding,
#         time_processor=MLP,
#         input_dim=32,
#         hidden_dims=[64, 128]
#     )
#     """
#     def __init__(
#         self,
#         mlp_class: Type[nn.Module],
#         time_embedding: Type[nn.Module],
#         time_processor: Type[nn.Module],
#         input_dim: int,
#         hidden_dims: Sequence[int]
#     ):
#         super().__init__()
#         self.time_embedding = time_embedding()
#         self.time_processor = time_processor()
#         self.mlp = mlp_class(hidden_dims=hidden_dims)
        
#         # 自动管理维度
#         self.proj = nn.LazyLinear(input_dim) if input_dim else nn.Identity()

#     def forward(self, x: torch.Tensor, time: torch.Tensor, training: bool = False) -> torch.Tensor:
#         # 时间特征处理
#         t_emb = self.time_embedding(time)
#         t_feat = self.time_processor(t_emb)
        
#         # 拼接特征
#         x = self.proj(x)
#         combined = torch.cat([x, t_feat], dim=-1)
        
#         # 通过MLP
#         return self.mlp(combined, training=training)

# # 测试用例
# if __name__ == "__main__":
#     # 测试MLP
#     batch_size = 32
#     input_dim = 128
#     test_input = torch.randn(batch_size, input_dim)
    
#     mlp = MLP(
#         hidden_dims=[256, 128, 64],
#         activations=nn.ReLU(),
#         activate_final=True,
#         layer_norm=True,
#         dropout_rate=0.1
#     )
    
#     output = mlp(test_input, training=True)
#     print(f"MLP输入形状: {test_input.shape}")
#     print(f"MLP输出形状: {output.shape}")
    
#     # 测试TimeMLP
#     class TimeEmbedding(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.embed = nn.Linear(1, 16)
            
#         def forward(self, time):
#             return self.embed(time)
    
#     time_mlp = TimeMLP(
#         mlp_class=MLP,
#         time_embedding=TimeEmbedding,
#         time_processor=MLP,
#         input_dim=128+16,  # 输入维度需要匹配
#         hidden_dims=[256, 128]
#     )
    
#     time_input = torch.randn(batch_size, 128)
#     time = torch.randn(batch_size, 1)
#     time_output = time_mlp(time_input, time)
#     print(f"\nTimeMLP输出形状: {time_output.shape}")