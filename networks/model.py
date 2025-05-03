# import os
# import collections
# from abc import ABC

# import flax
# import flax.linen as nn
# from flax import struct
# import jax
# import jax.numpy as jnp
# import optax

# from networks.types import Params, InfoDict
# from typing import Optional, Sequence, Tuple

# import orbax.checkpoint
# from flax.training import orbax_utils


# @struct.dataclass
# class Model(struct.PyTreeNode, ABC):
#     step: int
#     network: nn.Module = flax.struct.field(pytree_node=False)
#     params: Params
#     optimizer: Optional[optax.GradientTransformation] = flax.struct.field(
#         pytree_node=False)
#     opt_state: Optional[optax.OptState] = None
#     """
#     its network defines the computation tree for the inputs, it's a skeleton
#     its params is an independent object
#     its optimizer is defined from 'import optax | optax.adam'
#     its forward/backward functions are conducted using the customized apply_gradient function
#     use self.replace function modifies the  attributes
    
#     step: record the number of gradient updates
    
#     Model is initialized with Model.create() method
#     """

#     @classmethod
#     def create(cls,
#                network: nn.Module,
#                inputs: Sequence[jnp.ndarray],  # sample of inputs
#                optimizer: Optional[optax.GradientTransformation] = None,
#                clip_grad_norm: float = None) -> 'Model':
#         params = network.init(*inputs)  # (rng, other_inputs), params = {"params": ...}

#         if optimizer is not None:
#             if clip_grad_norm:
#                 optimizer = optax.chain(
#                     optax.clip_by_global_norm(max_norm=clip_grad_norm),
#                     optimizer)
#             opt_state = optimizer.init(params)
#         else:
#             opt_state = None

#         return cls(step=1,
#                    network=network,
#                    params=params,
#                    optimizer=optimizer,
#                    opt_state=opt_state
#                    )

#     def __call__(self, *args, **kwargs):
#         # the network defined by the jax nn.Model should be used by apply function with {'params': P} and other ..
#         return self.network.apply(self.params, *args, **kwargs)

#     def apply(self, *args, **kwargs):
#         return self.network.apply(*args, **kwargs)

#     def apply_gradient(self, loss_fn) -> Tuple['Model', InfoDict]:
#         grad_fn = jax.grad(loss_fn, has_aux=True)  # here auxiliary data is just the info dict
#         grads, info = grad_fn(self.params)

#         updates, new_opt_state = self.optimizer.update(grads, self.opt_state,
#                                                        self.params)

#         new_params = optax.apply_updates(self.params, updates)

#         return self.replace(step=self.step + 1,
#                             params=new_params,
#                             opt_state=new_opt_state), info  # gives new model, info

#     def get_state(self):
#         return {"step": self.step}

#     def save(self, save_path: str, force=True):
#         os.makedirs(os.path.dirname(save_path), exist_ok=force)
#         with open(save_path, 'wb') as f:
#             f.write(flax.serialization.to_bytes(self.params))

#     def load(self, load_path: str) -> 'Model':
#         # need to set a new_model = model.load(...)
#         with open(load_path, 'rb') as f:
#             params = flax.serialization.from_bytes(self.params, f.read())
#         return self.replace(params=params)

#     # def save(self, save_path: str, save_name: str = None):
#     #     if not os.path.exists(save_path):
#     #         os.makedirs(save_path)
#     #     save_args = orbax_utils.save_args_from_target(self)
#     #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer(checkpoint_name=save_name)
#     #     orbax_checkpointer.save(save_path, item=self, save_args=save_args, force=True)
#     #
#     # def load(self, load_path: str, save_name: str = None) -> 'Model':
#     #     """
#     #     This load require the correct definition of model structure.
#     #     """
#     #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer(checkpoint_name=save_name)
#     #     return orbax_checkpointer.restore(load_path, item=self)


# if __name__ == '__main__':
#     from networks.mlp import MLP

#     # from flax.training import checkpoints
#     # from flax.training import orbax_utils

#     net_ = MLP((256, 256))
#     optx_ = optax.adamw(learning_rate=1e-4)
#     obs = jnp.zeros((32, 12))
#     rng = jax.random.PRNGKey(1)

#     vars_ = net_.init(rng, obs)
#     optx_.init(vars_)

#     actor = Model.create(net_, (rng, obs), optx_)

#     # single save & load
#     # it must provide the correct structure

#     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     save_args = orbax_utils.save_args_from_target({"actor": actor, "data": jnp.ones((5,))})
#     orbax_checkpointer.save('ckpt/test', item={"actor": actor, "data": jnp.ones((5,))}, save_args=save_args, force=True)
#     target = {"actor": actor, "data": jnp.zeros((5,))}
#     raw_restored = orbax_checkpointer.restore('ckpt/test', item=target)
#     raw_restored
#     Model.create(raw_restored)

#     # training save
#     options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
#     checkpoint_manager = orbax.checkpoint.CheckpointManager(
#         '/tmp/flax_ckpt/orbax/managed', orbax_checkpointer, options)

#     save_args = orbax_utils.save_args_from_target(actor)
#     orbax_checkpointer.save('ckpt/test', actor, save_args=save_args, checkpoint_name='hi')

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Sequence, Tuple, Dict, Any

class Model(nn.Module):
    def __init__(self,
                 network: nn.Module,
                 optimizer: Optional[Optimizer] = None,
                 clip_grad_norm: Optional[float] = None):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.step = 1  # 初始化训练步数
        
    @classmethod
    def create(cls,
               network: nn.Module,
               optimizer_class: Optional[torch.optim.Optimizer] = None,
               optimizer_kwargs: Dict[str, Any] = None,
               clip_grad_norm: Optional[float] = None) -> 'Model':
        """
        创建模型实例
        :param network: PyTorch网络模块
        :param optimizer_class: 优化器类（如torch.optim.Adam）
        :param optimizer_kwargs: 优化器参数（如{"lr": 1e-3}）
        :param clip_grad_norm: 梯度裁剪阈值
        """
        # 初始化优化器
        optimizer = None
        if optimizer_class is not None:
            optimizer = optimizer_class(network.parameters(), **optimizer_kwargs)
            
        return cls(network, optimizer, clip_grad_norm)

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple['Model', Dict]:
        """
        执行梯度更新步骤
        :param loss_fn: 返回（损失值，信息字典）的函数
        """
        self.optimizer.zero_grad()
        
        # 计算损失和信息
        loss, info = loss_fn()
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                max_norm=self.clip_grad_norm
            )
        
        # 优化器更新
        self.optimizer.step()
        self.step += 1
        
        return self, info

    def get_state(self) -> Dict:
        return {
            "step": self.step,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None
        }

    def save(self, save_path: str):
        """保存完整模型状态"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "step": self.step,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None
        }, save_path)

    def load(self, load_path: str) -> 'Model':
        """加载模型状态"""
        checkpoint = torch.load(load_path)
        self.step = checkpoint["step"]
        self.network.load_state_dict(checkpoint["network_state"])
        if self.optimizer and checkpoint["optimizer_state"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return self

# 测试用例
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter
    
    # 定义示例网络
    class MLP(nn.Module):
        def __init__(self, hidden_dims):
            super().__init__()
            layers = []
            in_dim = 12
            for out_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
                in_dim = out_dim
            self.net = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.net(x)
    
    # 创建模型实例
    network = MLP([256, 256])
    model = Model.create(
        network=network,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-4},
        clip_grad_norm=1.0
    )
    
    # 示例训练步骤
    def dummy_loss_fn():
        x = torch.randn(32, 12)
        y = model(x)
        loss = y.mean()
        return loss, {"dummy_loss": loss.item()}
    
    # 执行梯度更新
    updated_model, info = model.apply_gradient(dummy_loss_fn)
    
    # 保存/加载测试
    model.save("ckpt/test.pth")
    loaded_model = Model.create(network, None).load("ckpt/test.pth")
