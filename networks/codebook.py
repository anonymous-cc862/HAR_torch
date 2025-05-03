
# import flax.linen as nn
# import jax
# import jax.numpy as jnp


# class CodeBook(nn.Module):
#     embedding_dim: int
#     num_codes: int

#     def setup(self):
#         self.codebook = self.param('codebook', nn.initializers.lecun_uniform(),
#                                    (self.num_codes, self.embedding_dim))  # (K, D)

#     def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#         l2_sum = jax.vmap(lambda x_: ((self.codebook - x_) ** 2).sum(axis=-1))(x)  # (B, z_dim) -> (B, num_ways)
#         codes = l2_sum.argmin(axis=-1)  # (B,)
#         code_vec = self.codebook[codes]  # (B, embedding_dim)

#         return code_vec, codes, l2_sum


import torch
import torch.nn as nn
import torch.nn.functional as F

class CodeBook(nn.Module):
    def __init__(self, embedding_dim: int, num_codes: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_codes = num_codes
        
        # 初始化codebook参数
        self.codebook = nn.Parameter(
            torch.empty(num_codes, embedding_dim)
        )
        self._init_weights()

    def _init_weights(self):
        # 对应JAX的lecun_uniform初始化
        # Lecun uniform实际上是limit = sqrt(3 / fan_in)
        fan_in = self.embedding_dim
        limit = (3.0 / fan_in) ** 0.5
        nn.init.uniform_(self.codebook, -limit, limit)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: 输入张量，形状为 (batch_size, embedding_dim)
        :return: 
            - code_vec: 最近的码本向量，形状 (batch_size, embedding_dim)
            - codes: 最近码本索引，形状 (batch_size,)
            - l2_sum: 所有码本的L2距离，形状 (batch_size, num_codes)
        """
        # 计算L2距离 (广播机制自动处理批量维度)
        diff = x.unsqueeze(1) - self.codebook.unsqueeze(0)  # (B, 1, D) - (1, K, D) -> (B, K, D)
        l2_sum = torch.sum(diff.pow(2), dim=-1)  # (B, K)

        # 找到最近码本索引
        codes = torch.argmin(l2_sum, dim=1)  # (B,)
        
        # 获取对应码本向量
        code_vec = self.codebook[codes]  # (B, D)

        return code_vec, codes, l2_sum

# 测试用例
if __name__ == "__main__":
    batch_size = 32
    embedding_dim = 64
    num_codes = 512
    
    # 初始化模块
    codebook = CodeBook(embedding_dim, num_codes)
    
    # 生成测试输入
    x = torch.randn(batch_size, embedding_dim)
    
    # 前向传播
    code_vec, codes, l2_sum = codebook(x)
    
    print("输入形状:", x.shape)            # torch.Size([32, 64])
    print("码本向量形状:", code_vec.shape)  # torch.Size([32, 64])
    print("索引形状:", codes.shape)        # torch.Size([32])
    print("距离矩阵形状:", l2_sum.shape)    # torch.Size([32, 512])
    
    # 验证最近邻选择是否正确
    # 随机选择一个样本验证
    sample_idx = 0
    min_code = codes[sample_idx].item()
    assert torch.allclose(code_vec[sample_idx], codebook.codebook[min_code])
    # print("最近邻验证通过!")