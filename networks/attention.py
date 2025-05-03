# import flax.linen as nn
# import jax.numpy as jnp
# from typing import Type


# def scaled_dot_product(q, k, v, mask=None):
#     d_k = q.shape[-1]
#     attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
#     attn_logits = attn_logits / jnp.sqrt(d_k)
#     if mask is not None:
#         attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
#     attention = nn.softmax(attn_logits, axis=-1)
#     values = jnp.matmul(attention, v)
#     return values, attention


# def expand_mask(mask):
#     assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
#     if mask.ndim == 3:
#         mask = mask.unsqueeze(1)
#     while mask.ndim < 4:
#         mask = mask.unsqueeze(0)
#     return mask


# class MultiHeadAttention(nn.Module):
#     # inner Q, K, V has dim = embed_dim/num_heads
#     # if embed_dim != input_dim, it gives cross attention
#     embed_dim: int  # Output dimension
#     num_heads: int  # Number of parallel heads (h)

#     @nn.compact
#     def __call__(self, x, mask=None):
#         # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
#         # Note that in many implementations you see "bias=False" which is optional
#         batch_size, seq_length, embed_dim = x.shape
#         assert embed_dim == self.embed_dim

#         if mask is not None:
#             mask = expand_mask(mask)
#         qkv = nn.Dense(3 * self.embed_dim,
#                        kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
#                        bias_init=nn.initializers.zeros  # Bias init with zeros
#                        )(x)  # qkv_proj

#         # Separate Q, K, V from linear output
#         qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)  # split the embed_dim = head*dims
#         qkv = qkv.transpose(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
#         q, k, v = jnp.array_split(qkv, 3, axis=-1)

#         # Determine value outputs
#         values, attention = scaled_dot_product(q, k, v, mask=mask)
#         values = values.transpose(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
#         values = values.reshape(batch_size, seq_length, embed_dim)
#         o = nn.Dense(self.embed_dim,
#                      kernel_init=nn.initializers.xavier_uniform(),
#                      bias_init=nn.initializers.zeros)(values)  # output projection

#         return o, attention


# class EncoderBlock(nn.Module):
#     input_dim: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
#     num_heads: int
#     mlp_dim: int
#     dropout_rate: float = 0.

#     @nn.compact
#     def __call__(self, x, mask=None, training=True):
#         # Attention part
#         attn_out, _ = MultiHeadAttention(embed_dim=self.input_dim,
#                                          num_heads=self.num_heads)(x, mask=mask)
#         x = x + nn.Dropout(self.dropout_rate)(attn_out, deterministic=not training)
#         x = nn.LayerNorm()(x)

#         # MLP part
#         linear_out = nn.Dense(self.mlp_dim)(x)
#         linear_out = nn.Dropout(self.dropout_rate)(linear_out, deterministic=not training)
#         linear_out = nn.relu(linear_out)
#         linear_out = nn.Dense(self.input_dim)(linear_out)

#         # add residual
#         x = x + nn.Dropout(self.dropout_rate)(linear_out, deterministic=not training)
#         x = nn.LayerNorm()(x)

#         return x


# class TransformerEncoder(nn.Module):
#     num_layers: int
#     input_dim: int
#     num_heads: int
#     mlp_dim: int
#     dropout_rate: float = 0

#     @nn.compact
#     def __call__(self, x, mask=None, training=True):
#         for _ in range(self.num_layers):
#             x = EncoderBlock(input_dim=self.input_dim,
#                              num_heads=self.num_heads,
#                              mlp_dim=self.mlp_dim,
#                              dropout_rate=self.dropout_rate)(x, mask=mask, training=training)
#         return x


# class TransformerEmbedding(nn.Module):
#     hidden_dim: int
#     time_embedding: Type[nn.Module]
#     time_net: Type[nn.Module]

#     @nn.compact
#     def __call__(self,
#                  x: jnp.ndarray,  # s = [obs, cond] if it's conditional  # (B, columns)
#                  time: jnp.ndarray,
#                  training: bool = False):

#         batch_size, num_columns = x.shape

#         t_ff = self.time_embedding()(time)
#         time_suffix = self.time_net()(t_ff, training=training)  # (B, time_dim)

#         # treat columns as sequence length in transformer
#         # add CLS token (B, columns) -> (B, 1 + columns)
#         x = jnp.concatenate([jnp.zeros((batch_size, 1)), x], axis=-1)
#         # add time embeddings (B, 1 + columns, 1) -> (B, 1 + columns, 1 + time_dim)
#         x = jnp.concatenate([jnp.expand_dims(x, axis=-1),
#                              time_suffix[:, jnp.newaxis, :].repeat(num_columns + 1, axis=1)], axis=-1)
#         # (B, 1 + columns, 1 + time_dim) -> (B, 1 + columns, D)
#         x = nn.Dense(self.hidden_dim)(x)

#         x = TransformerEncoder(num_layers=2,
#                                input_dim=self.hidden_dim,
#                                mlp_dim=64,
#                                num_heads=2,
#                                dropout_rate=0.1)(x, training=training)

#         return x[:, 0, :]  # fetch the embedding of the CLS token





# if __name__ == '__main__':
#     import jax
#     import jax.random as random

#     main_rng = jax.random.PRNGKey(1)
#     main_rng, x_rng = random.split(main_rng)
#     xx = random.normal(x_rng, (3, 16, 128))
#     # Create attention
#     mh_attn = MultiHeadAttention(embed_dim=128, num_heads=4)
#     # Initialize parameters of attention with random key and inputs
#     main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
#     params = mh_attn.init(init_rng, xx)
#     # Apply attention with parameters on the inputs
#     out, attn = mh_attn.apply(params, xx)
#     print('Out', out.shape, 'Attention', attn.shape)

#     f = EncoderBlock(input_dim=128, mlp_dim=256, num_heads=4, dropout_rate=0.1)
#     params = f.init({'params': init_rng, 'dropout': dropout_init_rng}, xx, training=True)
#     y = f.apply(params, xx, training=True, rngs={'dropout': random.key(2)})

#     f = TransformerEncoder(num_layers=5, input_dim=128, mlp_dim=256, num_heads=4, dropout_rate=0.1)
#     params = f.init({'params': init_rng, 'dropout': dropout_init_rng}, xx, training=True)
#     y = f.apply(params, xx, training=True, rngs={'dropout': random.key(2)})


#     from diffusion_feature import FourierFeatures, mish
#     from networks.mlp import MLP
#     from functools import partial

#     time_dim = 16

#     time_embedding = partial(FourierFeatures,
#                              output_size=time_dim,
#                              learnable=False)

#     time_net = partial(MLP,
#                        hidden_dims=(32, 32),
#                        activations=nn.leaky_relu,
#                        activate_final=False)

#     f = TransformerEmbedding(hidden_dim=12,  # should be the multiple of heads
#                              time_embedding=time_embedding,
#                              time_net=time_net)

#     xx = random.normal(x_rng, (64, 8))
#     params = f.init({'params': init_rng, 'dropout': dropout_init_rng},
#                     jnp.ones((64, 8)),
#                     jnp.ones((64, 1)),
#                     training=True)
#     y = f.apply(params, xx, jnp.ones((64, 1)), training=True, rngs={'dropout': random.key(2)})





import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Optional
from functools import partial

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / torch.sqrt(torch.tensor(d_k))
    
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

def expand_mask(mask):
    assert mask.dim() >= 2, "Mask must be at least 2-dimensional"
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)
        
    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape
        
        # Project QKV
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Compute attention
        values, attention = scaled_dot_product(q, k, v, mask)
        
        # Concatenate heads
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dim]
        values = values.contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # Output projection
        o = self.o_proj(values)
        return o, attention

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(mlp_dim, input_dim))
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x, mask=None):
        # Attention
        attn_out, _ = self.attention(x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # MLP
        linear_out = self.mlp(x)
        x = x + self.dropout2(linear_out)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, num_heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(input_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerEmbedding(nn.Module):
    def __init__(self, hidden_dim, time_embedding, time_net):
        super().__init__()
        self.time_embedding = time_embedding()
        self.time_net = time_net()
        self.cls_token = nn.Parameter(torch.zeros(1, 1))
        self.proj = nn.Linear(1 + time_net.hidden_dims[-1], hidden_dim)
        self.encoder = TransformerEncoder(
            num_layers=2,
            input_dim=hidden_dim,
            num_heads=2,
            mlp_dim=64,
            dropout_rate=0.1
        )
        
    def forward(self, x, time, training=False):
        batch_size, num_columns = x.shape
        
        # Time embedding
        t_ff = self.time_embedding(time)
        time_suffix = self.time_net(t_ff)
        
        # Add CLS token
        x = torch.cat([self.cls_token.expand(batch_size, -1), x], dim=1)
        
        # Combine features
        x = x.unsqueeze(-1)
        time_suffix = time_suffix.unsqueeze(1).expand(-1, num_columns+1, -1)
        x = torch.cat([x, time_suffix], dim=-1)
        
        # Projection
        x = self.proj(x)
        
        # Transformer encoder
        x = self.encoder(x)
        
        return x[:, 0, :]  # CLS token embedding

# 测试用例
if __name__ == "__main__":
    # 测试MultiHeadAttention
    x = torch.randn(3, 16, 128)
    mh_attn = MultiHeadAttention(embed_dim=128, num_heads=4)
    out, attn = mh_attn(x)
    print(f'MultiHeadAttention输出形状: {out.shape}, 注意力形状: {attn.shape}')

    # 测试EncoderBlock
    encoder_block = EncoderBlock(input_dim=128, num_heads=4, mlp_dim=256)
    y = encoder_block(x)
    print(f'EncoderBlock输出形状: {y.shape}')

    # 测试TransformerEncoder
    transformer = TransformerEncoder(num_layers=5, input_dim=128, num_heads=4, mlp_dim=256)
    y = transformer(x)
    print(f'TransformerEncoder输出形状: {y.shape}')

    # 测试TransformerEmbedding
    class FourierFeatures(nn.Module):
        def __init__(self, output_size=16, learnable=False):
            super().__init__()
            self.output_size = output_size
            
        def forward(self, x):
            return torch.randn(x.size(0), self.output_size)

    class MLP(nn.Module):
        def __init__(self, hidden_dims=(32,32)):
            super().__init__()
            self.hidden_dims = hidden_dims
            self.net = nn.Sequential(
                nn.Linear(16, hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LeakyReLU()
            )
            
        def forward(self, x):
            return self.net(x)

    emb = TransformerEmbedding(
        hidden_dim=12,
        time_embedding=partial(FourierFeatures, output_size=16),
        time_net=partial(MLP, hidden_dims=(32,32))
    )
    
    x = torch.randn(64, 8)
    time = torch.ones(64, 1)
    y = emb(x, time)
    # print(f'TransformerEmbedding输出形状: {y.shape}')