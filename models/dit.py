import math
import typing
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

def bias_dropout_add_scale(x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: torch.Tensor, residual: typing.Optional[torch.Tensor], prob: float, training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out

def get_bias_dropout_add_scale(training):

    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(x, bias, scale, residual, prob, training)
    return _bias_dropout_add

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

@torch.jit.script
def bias_dropout_add_scale_fused_train(x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: torch.Tensor, residual: typing.Optional[torch.Tensor], prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)

@torch.jit.script
def bias_dropout_add_scale_fused_inference(x: torch.Tensor, bias: typing.Optional[torch.Tensor], scale: torch.Tensor, residual: typing.Optional[torch.Tensor], prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)

@torch.jit.script
def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return (self.cos_cached, self.sin_cached)

def rotate_half(x):
    x1, x2 = (x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:])
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, :cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, :sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]

def residual_linear(x, W, x_skip, residual_scale):
    dim_out, dim_in = (W.shape[0], W.shape[1])
    return torch.addmm(x_skip.view(-1, dim_out), x.view(-1, dim_in), W.T, alpha=residual_scale).view(*x.shape[:-1], dim_out)

class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim, bias=True), nn.GELU(approximate='tanh'), nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c, seqlens=None, window_size=(-1, -1)):
        batch_size, seq_len = (x.shape[0], x.shape[1])
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=qkv.device)
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, seq_len, 0.0, causal=False, window_size=window_size)
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x

class EmbeddingLayer(nn.Module):

    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]

class DDitFinalLayer(nn.Module):

    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = Rotary(config.model.hidden_size // config.model.n_heads)
        self.algo = getattr(config, 'algo', None)
        if self.algo not in (None, 'recurrent'):
            raise ValueError('Expected MDLM or LoopMDM.')
        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout))
        self.blocks = nn.ModuleList(blocks)
        if self.algo == 'recurrent':
            self.head_layer_num = config.recurrent.head_layer_num
            self.tail_layer_num = config.recurrent.tail_layer_num
            self.mid_layer_num = config.model.n_blocks - self.head_layer_num - self.tail_layer_num
            if self.mid_layer_num <= 0:
                raise ValueError(f'mid_layer_num must be positive. Got n_blocks={config.model.n_blocks}, head_layer_num={self.head_layer_num}, tail_layer_num={self.tail_layer_num}')
            self.head_blocks = self.blocks[:self.head_layer_num]
            self.mid_blocks = self.blocks[self.head_layer_num:self.head_layer_num + self.mid_layer_num]
            self.tail_blocks = self.blocks[self.head_layer_num + self.mid_layer_num:]
            self.use_gradient_checkpointing = config.recurrent.get('use_gradient_checkpointing', False)
            self.use_loop_norm = config.recurrent.get('use_loop_norm', True)
            if self.use_loop_norm:
                self.loop_norm = LayerNorm(config.model.hidden_size)
        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma, mask_index=None, return_hidden=False, window_size=(-1, -1)):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None, window_size=window_size)
            hidden = x
            x = self.output_layer(x, c)
        if return_hidden:
            return (x, hidden)
        return x

    def _mid_loop_iteration(self, x, head_output, rotary_cos_sin, c, window_size):
        for block in self.mid_blocks:
            x = block(x, rotary_cos_sin, c, seqlens=None, window_size=window_size)
        x = x + head_output
        if self.use_loop_norm:
            x = self.loop_norm(x)
        return x

    def forward_recurrent(self, indices, sigma, num_loops, window_size=(-1, -1)):
        if self.algo != 'recurrent':
            raise RuntimeError("forward_recurrent called but algo is not 'recurrent'")
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for block in self.head_blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None, window_size=window_size)
            head_output = x
            for loop_idx in range(num_loops):
                if self.use_gradient_checkpointing and self.training:
                    x = torch_checkpoint(self._mid_loop_iteration, x, head_output, rotary_cos_sin, c, window_size, use_reentrant=False)
                else:
                    x = self._mid_loop_iteration(x, head_output, rotary_cos_sin, c, window_size)
            for block in self.tail_blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None, window_size=window_size)
            x = self.output_layer(x, c)
        return x
