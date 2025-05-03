
# import flax
# import collections
# from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

# Batch = collections.namedtuple(
#     'Batch',
#     ['X', 'num_idx', 'cate_idx', 'y'])

# MetaBatch = collections.namedtuple(
#     'MetaBatch',
#     ['X', 'X_tar', 'X_support'])

# PRNGKey = Any
# Params = flax.core.FrozenDict[str, Any]
# Shape = Sequence[int]
# InfoDict = Dict[str, Any]


from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union
import torch

@dataclass(frozen=True)
class Batch:
    """数据批次结构"""
    X: torch.Tensor           # 输入特征
    num_idx: torch.Tensor     # 数值特征索引
    cate_idx: torch.Tensor    # 类别特征索引
    y: Optional[torch.Tensor] = None  # 目标标签（可选）

@dataclass(frozen=True)
class MetaBatch:
    """元学习批次结构"""
    X: torch.Tensor           # 常规输入
    X_tar: torch.Tensor       # 目标域输入
    X_support: torch.Tensor   # 支持集输入

# 类型别名定义
Shape = Sequence[int]         # 张量形状类型
InfoDict = Dict[str, Any]     # 信息字典类型
Params = Dict[str, torch.Tensor]  # 参数字典类型（对应PyTorch的state_dict）
