import itertools
import json
import math
import os
import typing
from dataclasses import dataclass
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor
import omegaconf
import dataloader
import models
import noise_schedule
import utils
LOG2 = math.log(2)

@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor

class NLL(torchmetrics.aggregation.MeanMetric):
    pass

class BPD(NLL):

    def compute(self) -> Tensor:
        return self.mean_value / self.weight / LOG2

class Perplexity(NLL):

    def compute(self) -> Tensor:
        return torch.exp(self.mean_value / self.weight)

class Diffusion(L.LightningModule):

    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.antithetic_sampling = self.config.training.antithetic_sampling
        if not hasattr(self.tokenizer, 'mask_token') or self.tokenizer.mask_token is None:
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = self.tokenizer.mask_token_id
        self.parameterization = self.config.parameterization
        self.backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        self.softplus = torch.nn.Softplus()
        metrics = torchmetrics.MetricCollection({'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='train/')
        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(itertools.chain(self.backbone.parameters(), self.noise.parameters()), decay=self.config.training.ema)
        else:
            self.ema = None
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self.algo = getattr(self.config, 'algo', None)
        if self.algo == 'recurrent':
            self.max_loop = self.config.recurrent.max_loop
        self._validate_configuration()

    def _validate_configuration(self):
        if self.config.backbone != 'dit' or self.algo not in (None, 'recurrent'):
            raise ValueError('Expected a DiT MDLM or LoopMDM configuration.')
        if self.config.parameterization != 'subs' or self.config.T != 0:
            raise ValueError('Expected continuous-time SUBS parameterization.')
        if self.algo == 'recurrent' and self.max_loop < 1:
            raise ValueError('max_loop must be positive.')

    def _sample_loop_count(self):
        return torch.randint(1, self.max_loop + 1, (1,)).item()

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['total']['completed'] = checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] = checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['current']['completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop']['epoch_loop.state_dict']['_batches_that_stepped'] = checkpoint['loops']['fit_loop']['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        distributed = self.trainer._accelerator_connector.use_distributed_sampler and self.trainer._accelerator_connector.is_distributed
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if distributed and self.fast_forward_epochs is not None and (self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({'epoch': self.fast_forward_epochs, 'counter': self.fast_forward_batches * self.config.loader.batch_size})
            updated_dls.append(torch.utils.data.DataLoader(dl.dataset, batch_size=self.config.loader.batch_size, num_workers=self.config.loader.num_workers, pin_memory=self.config.loader.pin_memory, sampler=dl_sampler, drop_last=getattr(dl, 'drop_last', False), collate_fn=getattr(dl, 'collate_fn', None), shuffle=False, persistent_workers=self.config.loader.num_workers > 0))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters(), self.noise.parameters()))

    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma, num_loops=None, return_hidden=False):
        sigma = self._process_sigma(sigma)
        hidden = None
        window_size = (-1, -1)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            if self.algo == 'recurrent':
                if num_loops is None:
                    num_loops = self._sample_loop_count()
                logits = self.backbone.forward_recurrent(x, sigma, num_loops=num_loops, window_size=window_size)
            elif return_hidden:
                logits, hidden = self.backbone(x, sigma, return_hidden=True, window_size=window_size)
            else:
                logits = self.backbone(x, sigma, window_size=window_size)
        logits = self._subs_parameterization(logits=logits, xt=x)
        if return_hidden:
            return (logits, hidden)
        return logits

    def _compute_loss(self, batch):
        losses = self._loss(batch['input_ids'], batch.get('attention_mask'))
        self.train_metrics.update(losses.nlls, losses.token_mask)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)
        return losses.loss

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log(name='trainer/loss', value=loss.item(), on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        param_groups = itertools.chain(self.backbone.parameters(), self.noise.parameters())
        optimizer = torch.optim.AdamW(param_groups, lr=self.config.optim.lr, betas=(self.config.optim.beta1, self.config.optim.beta2), eps=self.config.optim.eps, weight_decay=self.config.optim.weight_decay)
        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {'scheduler': scheduler, 'interval': 'step', 'name': 'trainer/lr'}
        return ([optimizer], [scheduler_dict])

    def on_before_optimizer_step(self, optimizer):
        total_norm = 0.0
        num_nan = 0
        num_inf = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                nan_mask = torch.isnan(param.grad)
                inf_mask = torch.isinf(param.grad)
                if nan_mask.any():
                    num_nan += nan_mask.sum().item()
                    param.grad[nan_mask] = 0.0
                if inf_mask.any():
                    num_inf += inf_mask.sum().item()
                    param.grad[inf_mask] = 0.0
                param.grad.clamp_(-1000.0, 1000.0)
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if self.global_step % 100 == 0:
            self.log('trainer/grad_norm', total_norm, on_step=True, on_epoch=False)
            if num_nan > 0:
                self.log('trainer/grad_nan_count', num_nan, on_step=True, on_epoch=False)
            if num_inf > 0:
                self.log('trainer/grad_inf_count', num_inf, on_step=True, on_epoch=False)

    def q_xt(self, x, move_chance):
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t_min, t_max = (self.sampling_eps, 1.0)
        t = t_min + (t_max - t_min) * _eps_t
        return t

    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        if seqlen > self.config.model.length:
            assert seqlen == 2 * self.config.model.length
            start = np.random.choice(self.config.model.length)
            end = start + self.config.model.length
            input_tokens = x0[:, start:end]
            output_tokens = x0[:, start + 1:end + 1]
            new_attention_mask = attention_mask[:, start:end]
            input_tokens[:, 0] = self.tokenizer.bos_token_id
            output_tokens[:, -1] = self.tokenizer.eos_token_id
        else:
            input_tokens = x0
            output_tokens = None
            new_attention_mask = attention_mask
        return (input_tokens, output_tokens, new_attention_mask)

    def _get_noise_fn(self):
        return self.noise

    def _forward_pass_diffusion(self, x0):
        t = self._sample_t(x0.shape[0], x0.device)
        noise_fn = self._get_noise_fn()
        sigma, dsigma = noise_fn(t)
        unet_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])
        xt = self.q_xt(x0, move_chance)
        num_loops = None
        model_output = self.forward(xt, unet_conditioning, num_loops=num_loops)
        utils.print_nans(model_output, 'model_output')
        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
        loss = -log_p_theta * (dsigma / torch.expm1(sigma).clamp(min=1e-06))[:, None]
        return loss

    def _loss(self, x0, attention_mask):
        input_tokens, output_tokens, attention_mask = self._maybe_sub_sample(x0, attention_mask)
        loss = self._forward_pass_diffusion(input_tokens)
        nlls = loss * attention_mask
        count = attention_mask.sum()
        batch_nll = nlls.sum()
        token_nll = batch_nll / count
        return Loss(loss=token_nll, nlls=nlls, token_mask=attention_mask)
