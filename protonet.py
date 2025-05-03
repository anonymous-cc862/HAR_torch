# import jax
# import jax.numpy as jnp
# from functools import partial
# from networks.types import Callable, Batch


# @partial(jax.jit, static_argnames=('adaptive_steps', 'temperature'))
# def jit_soft_kmeans(query_embedding: jnp.ndarray,  # (B, embed_dim)
#                     c0: jnp.ndarray,  # (n_way x num_shot, embed_dim)
#                     adaptive_steps: int,
#                     temperature: float
#                     ):
#     """
#     it returns instance-specific labels: indicating the [label]-th item of c0
#     """
#     def adaptive_proto_fn(ct, _):  # (n_way, embed_dim)
#         logit = jax.vmap(lambda z: jnp.exp(-((z - ct) ** 2/temperature).sum(axis=-1)))(query_embedding)  # (B, embed_dim) -> (B, n_way)
#         confid = logit / logit.sum(axis=-1, keepdims=True)  # (B, n_way) confidence weight matrix, normed along n_way
#         sum_qf = jax.vmap(lambda q: (q[:, jnp.newaxis] * query_embedding).sum(axis=0))(confid.transpose())  # (n_way, embed_dim)
#         c_t_new = (ct + sum_qf) / (1 + confid.transpose().sum(axis=-1, keepdims=True))
#         return c_t_new, ()  # (n_way, embed_dim)
#     proto_embeddings, () = jax.lax.scan(adaptive_proto_fn,
#                                         c0,
#                                         jnp.arange(adaptive_steps, 0, -1),
#                                         unroll=5)
#     scores = jax.vmap(lambda z: jnp.exp(-((z - proto_embeddings) ** 2/temperature).sum(axis=-1)))(query_embedding)
#     labels = scores.argmax(axis=-1)

#     return scores, labels


# def iterative_protonet_predict(embed_fn: Callable,
#                                support_set: Batch,
#                                n_way: int,
#                                num_shot: int,
#                                query_x: jnp.array,
#                                # unlabeled_x: jnp.array,
#                                iteration_steps,
#                                generator: Callable = None,
#                                num_samples: int = 10,
#                                temperature: float = 1,
#                                mean_center: bool = False) -> tuple[jnp.array, jnp.array]:

#     assert all(jnp.sort(support_set.y) == support_set.y), "The support set should be in order of [0,1,2,...]"
#     query_num = query_x.shape[0]
#     query_embed = embed_fn(query_x)  # (B, z_dim)
#     # query_embed = embed_fn(jnp.concatenate([query_x, unlabeled_x], axis=0))  # (B, z_dim)
#     s_embed = embed_fn(support_set.X)  # (num_shot x n_way, z_dim) [0,0,0, 1,1,1, 2,2,2, ...] (n_way, num_shot,-1)

#     c0 = s_embed
#     # if generator is None:
#     #     c0 = s_embed  # .reshape(n_way, num_shot, -1).mean(axis=1)  # requires proper order of support samples
#     if generator is not None:
#         rep_s_embed = jnp.repeat(s_embed, num_samples, axis=0)
#         x0 = generator(rep_s_embed)  # (n_way x num_shot x num_samples)
#         x0_embed = embed_fn(x0)
#         query_embed = jnp.concatenate([query_embed, x0_embed], axis=0)  # expand query embeddings
#         # x0_embed = embed_fn(x0).reshape(n_way, num_shot * num_samples, -1)
#         # c0 = jnp.concatenate([x0_embed, s_embed.reshape(n_way, num_shot, -1)], axis=1)  # .mean(axis=1)

#     if mean_center:
#         c0 = c0.reshape(n_way, num_shot, -1).mean(axis=1)
#         y = jnp.arange(n_way)
#     else:
#         y = support_set.y.repeat((num_samples + 1))  # (num_samples + 1) * num_shot

#     scores, labels = jit_soft_kmeans(query_embed,
#                                      c0,
#                                      iteration_steps,
#                                      temperature,
#                                      )

#     return scores, y[labels][:query_num]




# def diffusion_protonet_predict(embed_fn: Callable,
#                                support_set: Batch,
#                                n_way: int,
#                                query_x: jnp.array,
#                                generator: Callable,
#                                num_samples: int = 10) -> tuple[jnp.array, jnp.array]:
#     # TODO: generate more support samples
#     num_shot = sum(support_set.y == 0)
#     query_embed = embed_fn(query_x)  # (B, z_dim)
#     s_embed = embed_fn(support_set.X)  # (num_shot x n_way, z_dim) [0,0,0, 1,1,1, 2,2,2, ...] (n_way, num_shot,-1)
#     # (n_way x num_shot x rep,-1) can be reshaped to (n_way, num_shot x rep,-1)
#     rep_s_embed = jnp.repeat(s_embed, num_samples, axis=0)
#     x0 = generator(rep_s_embed)
#     # rng, x0 = decoder(rng, noise_model_fn, noise_model_params, rep_s_embed)
#     x0_embed = embed_fn(x0).reshape(n_way, num_shot * num_samples, -1)
#     # concatenate samples embeddings and support embeddings
#     c0 = jnp.concatenate([x0_embed, s_embed.reshape(n_way, num_shot, -1)], axis=1).mean(axis=1)
#     scores = jax.vmap(lambda z: jnp.exp(-((z - c0) ** 2).sum(axis=-1)))(query_embed)
#     labels = scores.argmax(axis=-1)

#     return scores, labels


# import torch
# from typing import Callable, Tuple
# from dataclasses import dataclass
# import numpy as np

# @dataclass
# class Batch:
#     X: torch.Tensor
#     y: torch.Tensor
#     num_idx: torch.Tensor
#     cate_idx: torch.Tensor
# # @dataclass(frozen=False)
# # class Batch:
# #     X: torch.Tensor
# #     y: torch.Tensor
# #     num_idx: torch.Tensor
# #     cate_idx: torch.Tensor


# def jit_soft_kmeans(query_embedding: torch.Tensor,  # (B, embed_dim)
#                 c0: torch.Tensor,              # (n_way x num_shot, embed_dim)
#                 adaptive_steps: int,
#                 temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     PyTorch版本的自适应原型网络
#     """
#     proto_embeddings = c0.clone()
    
#     for _ in range(adaptive_steps):
#         # 计算相似度得分 (B, n_way)
#         logits = torch.exp(-torch.cdist(query_embedding, proto_embeddings, p=2).pow(2)) / temperature
        
#         # 计算置信权重
#         confid = logits / logits.sum(dim=1, keepdim=True)  # (B, n_way)
        
#         # 更新原型
#         weighted_sum = torch.mm(confid.T, query_embedding)  # (n_way, embed_dim)
#         sum_weights = confid.sum(dim=0).unsqueeze(1)        # (n_way, 1)
#         proto_embeddings = (proto_embeddings + weighted_sum) / (1 + sum_weights)
    
#     # 计算最终得分和标签
#     scores = torch.exp(-torch.cdist(query_embedding, proto_embeddings, p=2).pow(2))
#     labels = scores.argmax(dim=1)
    
#     return scores, labels

# def iterative_protonet_predict(
#     embed_fn: Callable,
#     support_set: Batch,
#     n_way: int,
#     num_shot: int,
#     query_x: torch.Tensor,
#     iteration_steps: int,
#     generator: Callable = None,
#     num_samples: int = 10,
#     temperature: float = 1.0,
#     mean_center: bool = False
# ) -> Tuple[torch.Tensor, torch.Tensor]:
    
#     # Ensure tensor types
#     if isinstance(support_set.y, np.ndarray):
#         support_set.y = torch.from_numpy(support_set.y).long()
#     if isinstance(support_set.X, np.ndarray):
#         support_set.X = torch.from_numpy(support_set.X).float()

#     # 确保支持集标签有序
#     assert torch.all(torch.sort(support_set.y)[0] == support_set.y), "支持集需要按顺序排列"
    
#     # 获取查询嵌入
#     query_embed = embed_fn(query_x)  # (B, z_dim)
#     query_num = query_x.size(0)
    
#     # 处理支持集嵌入
#     s_embed = embed_fn(support_set.X)  # (n_way*num_shot, z_dim)
    
#     # 使用生成器扩展支持集
#     c0 = s_embed
#     if generator is not None:
#         # 生成合成样本
#         rep_s_embed = s_embed.repeat(num_samples, 1)
#         x0 = generator(rep_s_embed)
#         x0_embed = embed_fn(x0)
#         query_embed = torch.cat([query_embed, x0_embed], dim=0)
    
#     # 初始化原型
#     if mean_center:
#         c0 = s_embed.view(n_way, num_shot, -1).mean(dim=1)
#         y = torch.arange(n_way, device=query_embed.device)
#     else:
#         y = support_set.y.repeat(num_samples + 1)
    
#     # 执行软k-means聚类
#     scores, labels = jit_soft_kmeans(
#         query_embed=query_embed,
#         c0=c0,
#         adaptive_steps=iteration_steps,
#         temperature=temperature
#     )
    
#     return scores[:query_num], y[labels][:query_num]

# def diffusion_protonet_predict(
#     embed_fn: Callable,
#     support_set: Batch,
#     n_way: int,
#     query_x: torch.Tensor,
#     generator: Callable,
#     num_samples: int = 10
# ) -> Tuple[torch.Tensor, torch.Tensor]:
    
#     num_shot = torch.sum(support_set.y == 0).item()
    
#     # 获取查询嵌入
#     query_embed = embed_fn(query_x)
    
#     # 处理支持集嵌入
#     s_embed = embed_fn(support_set.X)
    
#     # 生成合成样本
#     rep_s_embed = s_embed.repeat(num_samples, 1)
#     x0 = generator(rep_s_embed)
#     x0_embed = embed_fn(x0)
    
#     # 重塑并合并嵌入
#     x0_embed = x0_embed.view(n_way, num_shot * num_samples, -1)
#     s_embed = s_embed.view(n_way, num_shot, -1)
#     c0 = torch.cat([x0_embed, s_embed], dim=1).mean(dim=1)
    
#     # 计算相似度得分
#     scores = torch.exp(-torch.cdist(query_embed, c0, p=2).pow(2))
#     labels = scores.argmax(dim=1)
    
#     return scores, labels

import torch
import numpy as np
from typing import Callable, Tuple
from dataclasses import dataclass

@dataclass(frozen=False)
class Batch:
    X: torch.Tensor
    y: torch.Tensor
    num_idx: torch.Tensor
    cate_idx: torch.Tensor

def to_tensor(x, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    elif not torch.is_tensor(x):
        return torch.tensor(x, dtype=dtype)
    return x

def jit_soft_kmeans(query_embedding: torch.Tensor,  # (B, embed_dim)
                    c0: torch.Tensor,              # (n_way x num_shot, embed_dim)
                    adaptive_steps: int,
                    temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch版本的自适应原型网络
    """
    query_embedding = to_tensor(query_embedding)
    c0 = to_tensor(c0)
    proto_embeddings = c0.clone()

    for _ in range(adaptive_steps):
        logits = torch.exp(-torch.cdist(query_embedding, proto_embeddings, p=2).pow(2)) / temperature
        confid = logits / logits.sum(dim=1, keepdim=True)
        weighted_sum = torch.mm(confid.T, query_embedding)
        sum_weights = confid.sum(dim=0).unsqueeze(1)
        proto_embeddings = (proto_embeddings + weighted_sum) / (1 + sum_weights)

    scores = torch.exp(-torch.cdist(query_embedding, proto_embeddings, p=2).pow(2))
    labels = scores.argmax(dim=1)

    return scores, labels

def iterative_protonet_predict(
    embed_fn: Callable,
    support_set: Batch,
    n_way: int,
    num_shot: int,
    query_x: torch.Tensor,
    iteration_steps: int,
    generator: Callable = None,
    num_samples: int = 10,
    temperature: float = 1.0,
    mean_center: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 确保所有字段是 tensor
    support_set = Batch(
        X=to_tensor(support_set.X, dtype=torch.float32),
        y=to_tensor(support_set.y, dtype=torch.long),
        num_idx=to_tensor(support_set.num_idx, dtype=torch.long),
        cate_idx=to_tensor(support_set.cate_idx, dtype=torch.long),
    )

    # 确保支持集标签有序
    assert torch.all(torch.sort(support_set.y)[0] == support_set.y), "支持集需要按顺序排列"

    query_x = to_tensor(query_x, dtype=torch.float32)
    query_embed = embed_fn(query_x)
    query_num = query_x.size(0)

    s_embed = embed_fn(support_set.X)
    c0 = s_embed

    if generator is not None:
        rep_s_embed = s_embed.repeat(num_samples, 1)
        x0 = generator(rep_s_embed)
        x0_embed = embed_fn(x0)
        query_embed = torch.cat([query_embed, x0_embed], dim=0)

    if mean_center:
        c0 = s_embed.view(n_way, num_shot, -1).mean(dim=1)
        y = torch.arange(n_way, device=query_embed.device)
    else:
        y = support_set.y.repeat(num_samples + 1)

    scores, labels = jit_soft_kmeans(
        query_embedding=query_embed,
        c0=c0,
        adaptive_steps=iteration_steps,
        temperature=temperature
    )

    return scores[:query_num], y[labels][:query_num]

def diffusion_protonet_predict(
    embed_fn: Callable,
    support_set: Batch,
    n_way: int,
    query_x: torch.Tensor,
    generator: Callable,
    num_samples: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 确保所有字段是 tensor
    support_set = Batch(
        X=to_tensor(support_set.X, dtype=torch.float32),
        y=to_tensor(support_set.y, dtype=torch.long),
        num_idx=to_tensor(support_set.num_idx, dtype=torch.long),
        cate_idx=to_tensor(support_set.cate_idx, dtype=torch.long),
    )
    query_x = to_tensor(query_x, dtype=torch.float32)

    num_shot = torch.sum(support_set.y == 0).item()

    query_embed = embed_fn(query_x)
    s_embed = embed_fn(support_set.X)

    rep_s_embed = s_embed.repeat(num_samples, 1)
    x0 = generator(rep_s_embed)
    x0_embed = embed_fn(x0)

    x0_embed = x0_embed.view(n_way, num_shot * num_samples, -1)
    s_embed = s_embed.view(n_way, num_shot, -1)
    c0 = torch.cat([x0_embed, s_embed], dim=1).mean(dim=1)

    scores = torch.exp(-torch.cdist(query_embed, c0, p=2).pow(2))
    labels = scores.argmax(dim=1)

    return scores, labels
