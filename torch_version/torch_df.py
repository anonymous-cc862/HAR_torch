import torch
import math
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch_version.networks import MLP#networks import MLP#
import collections


EPS = 1e-5


Batch = collections.namedtuple(
    'Batch',
    ['X', 'num_idx', 'cate_idx', 'y'])


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionFeatureExtractor(object):

    def __init__(
        self,
        seed,
        x_dim: int,
        x_cat_counts: list,  # given by [num_cats, ...]
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        k_way: int,
        alpha: float,
        std_scale: np.ndarray,
        col_select_ratio: float = 0.1,
        embed_t_dependent=True,
        lr: float = 3e-4,
        no_diffusion=False, dropout_rate=None, layer_norm=False,
        T: int = 10,  # number of backward steps
        num_last_repeats: int = 0,
        time_dim: int = 16,
        beta_schedule: str = "vp",
        #lr_decay_steps: int = 100000,
        sampler: str = "ddpm",
        temperature: float = 1,
        # device: str = "cpu",
    ):

        self.x_dim = x_dim
        self.x_cat_counts = x_cat_counts
        self.embed_dim = embed_dim
        self.std_scale = std_scale
        self.col_select_ratio = col_select_ratio
        self.num_proj_scale = np.sqrt(3 * self.col_select_ratio * (1 - self.col_select_ratio))
        self.k_way = k_way
        self.rdm_weight = alpha

        self.noise_model = MLP(x_dim + embed_dim + time_dim, x_dim, hidden_dim, num_layers)
        self.embed_model = MLP(x_dim, embed_dim, hidden_dim, num_layers)
        self.time_embed = SinusoidalPosEmb(dim=time_dim)
        
        self.noise_opt = torch.optim.Adam(self.noise_model.parameters(), lr=lr)
        self.embed_opt = torch.optim.Adam(self.embed_model.parameters(), lr=lr)

        """
        DDPM parameters
        """
        """
        βₜ
        """
        self.denoising_steps = T#denoising_steps
        self.betas = cosine_beta_schedule(T)#denoising_steps)
        """
        αₜ = 1 - βₜ
        """
        self.alphas = 1.0 - self.betas
        """
        α̅ₜ= ∏ᵗₛ₌₁ αₛ 
        """
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        """
        α̅ₜ₋₁
        """
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        """
        √ α̅ₜ
        """
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        """
        √ 1-α̅ₜ
        """
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        """
        √ 1\α̅ₜ
        """
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        """
        √ 1\α̅ₜ-1
        """
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        """
        β̃ₜ = σₜ² = βₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)
        """
        self.ddpm_var = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        """
        μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
        """
        self.ddpm_mu_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.ddpm_mu_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        
        self._n_training_steps = 0
            
    def update(self, batch: Batch):
        
        with torch.no_grad():
        
            batch_size, x_dim = batch.X.shape
            n_num_feature, n_cat_feature = len(batch.num_idx), len(batch.cate_idx)
            t = torch.randint(0, self.denoising_steps, (batch_size, 1))
            sigma = torch.Tensor(self.std_scale[np.newaxis, :].repeat(batch_size, axis=0))
            eps_sample = torch.randn(batch.X.shape)
            perm_idx = (torch.arange(batch_size) - 1) % batch_size 
            
            W1 = torch.rand((n_num_feature, n_num_feature))*2*self.num_proj_scale-self.num_proj_scale
            W2 = torch.bernoulli(torch.ones((n_cat_feature, n_cat_feature))*self.col_select_ratio)
            W1, W2 = W1/np.sqrt(n_num_feature + EPS), W2/np.sqrt(n_cat_feature + EPS)
            
            num_map, cat_map = torch.Tensor(batch.X[:, batch.num_idx]) @ W1, torch.Tensor(batch.X[:, batch.cate_idx]) @ W2
            rand_proj = torch.concatenate([num_map, cat_map], axis=-1)
            normed_proj = rand_proj / (torch.norm(rand_proj, dim=-1, keepdim=True) + EPS)
            xt = self.sqrt_alphas_cumprod[t] * torch.Tensor(batch.X) + self.sqrt_one_minus_alphas_cumprod[t] * eps_sample
            
            
        # training
        self.noise_opt.zero_grad()
        self.embed_opt.zero_grad()  
            
        embedding = self.embed_model(torch.Tensor(batch.X)+ sigma * torch.randn(batch.X.shape))
        pred_eps = self.noise_model(torch.concatenate([xt, self.time_embed(t.flatten()), embedding], axis=-1))
        
        with torch.no_grad():
            embedding_norm = torch.norm(embedding, dim=-1, keepdim=True)
        
        normed_embed = embedding / (embedding_norm+ EPS)
        
        # use normed L2 == cosine
        embed_dist = ((normed_embed - normed_embed[perm_idx]) ** 2).sum(axis=-1)
        rand_dist = ((normed_proj - normed_proj[perm_idx]) ** 2).sum(axis=-1)

        # random distance prediction loss
        rdm_loss = ((embed_dist - rand_dist) ** 2).mean()

        noise_loss = ((pred_eps - eps_sample) ** 2).sum(axis=-1).mean()
        
        loss = noise_loss + self.rdm_weight * rdm_loss
        
        loss.backward()
        self.embed_opt.step()
        self.noise_opt.step()
        
        self._n_training_steps += 1
        
        return {'loss': loss.item(),
                'noise_loss': noise_loss.item(),
                'rdm_loss': rdm_loss.item(),
                }
            

    def get_embedding(self, x):
        with torch.no_grad():
            return self.embed_model(torch.Tensor(x))

    def generate(self, embedding):
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        batch_size = embedding.shape[0]
        noise = torch.randn((batch_size, self.x_dim))
        # TODO: implement sampling logic
        return noise.detach().cpu().numpy()    
    
    
    
    
if __name__ == "__main__":
    from datasets import OpenmlDataset
    from tqdm import trange
    data = OpenmlDataset("opt", one_hot=True, norm="minmax")
    feature_std = data.X_stds
    
    self  = DiffusionFeatureExtractor(x_dim=len(data.all_features),
                                    x_cat_counts=[],  # len(data.cat_features),
                                    embed_dim=10,
                                    hidden_dim=256,
                                    num_layers=3,
                                    k_way=data.num_classes,
                                    alpha=1,
                                    #rdm_weight=1,
                                    std_scale=data.X_stds,
                                    T=10,
                                    #denoising_steps=10,
                                    col_select_ratio=0.1,
                                    )
    
    stats = {}
    
    for _ in trange(1000):
        batch = data.sample(256)
        info = self.update(batch)
        
        for k_, v_ in info.items():
            if k_ not in stats:
                stats[k_] = [v_]
            else:
                stats[k_].append(v_)
                
                
    from matplotlib import pyplot as plt
    
    for k,v in stats.items():
        plt.plot(v)
        plt.title(k)
        plt.show()
        
    # get feature:
    
    self.get_embedding(batch.X)
        
    
    
