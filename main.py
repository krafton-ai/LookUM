import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import dataloader
import diffusion
import utils
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

@L.pytorch.utilities.rank_zero_only
def _print_config(config: omegaconf.DictConfig, resolve: bool=True, save_cfg: bool=True) -> None:
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)
    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_section, resolve=resolve)
        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open('{}/config_tree.txt'.format(config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)

def _train(config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(config=omegaconf.OmegaConf.to_object(config), **config.wandb)
    if config.checkpointing.resume_from_ckpt and config.checkpointing.resume_ckpt_path is not None and utils.fsspec_exists(config.checkpointing.resume_ckpt_path):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    train_ds = dataloader.get_train_dataloader(config, tokenizer)
    model = diffusion.Diffusion(config, tokenizer=tokenizer)
    trainer = hydra.utils.instantiate(config.trainer, default_root_dir=os.getcwd(), callbacks=callbacks, strategy=hydra.utils.instantiate(config.strategy), logger=wandb_logger)
    trainer.fit(model, train_dataloaders=train_ds, ckpt_path=ckpt_path)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    if config.backbone != 'dit' or config.algo not in (None, 'recurrent'):
        raise ValueError('Expected a DiT MDLM or LoopMDM configuration.')
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
    _train(config, logger, tokenizer)
if __name__ == '__main__':
    main()
